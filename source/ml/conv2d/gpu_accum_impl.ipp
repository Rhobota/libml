/*
 * Note: CNNs has a concept of "kernel" and CUDA has a concept of "kernel". That could
 * cause confusion. In this file we will only talk about CNN kernels, and we will avoid
 * using the term "kernel" to talk about that CUDA concept--we will use alternate words
 * for that instead.
 */


/*
 * If you change BLOCK_SIZE_X or BLOCK_SIZE_Y or WARP_SIZE, then
 * this value will need to change.
 *
 * This value is used to cause access to the input shared memory
 * array to be coalesced (making the code run MUCH faster).
 */
#define INPUT_COALESCED_EXTRA_PADDING 16


namespace ml
{
namespace conv2d
{
namespace gpu
{


#if GPU_ACCUM_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_ACCUM_USE_TEMPLATE
accum_in_one_pass_templated(
#else
accum_in_one_pass(
#endif
#if !GPU_ACCUM_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,
              fml* dk_ptr,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    // We use shared memory so that each global memory value only must be read once!
    // Makes everything much much faster.
    // We will keep all components of the input block in shared memory at the same time.
    // And we will also keep ONE component of the dA block in shared memory at a time.
    extern __shared__ fml memory_shared[];
    fml* input_shared = memory_shared + 0;  // <--v-- note the weird dimensions on input_shared; this is so that shared memory access is coalesced; be sure to index this array properly!
    fml* dA_shared    = input_shared + (BLOCK_SIZE_Y*(BLOCK_SIZE_X+KERNEL_COLS)+INPUT_COALESCED_EXTRA_PADDING+KERNEL_ROWS*KERNEL_COLS)*INPUT_COMPONENTS;

    // Useful to have:
    bool isOutputThread;
    {
        i32 global_y = blockIdx.y * (BLOCK_SIZE_Y-KERNEL_ROWS+1) + threadIdx.y;  global_y -= KERNEL_ROWS/2;
        i32 global_x = blockIdx.x * (BLOCK_SIZE_X-KERNEL_COLS+1) + threadIdx.x;  global_x -= KERNEL_COLS/2;
        bool isInsideInput = (global_y >= 0) & (global_y < inputRows) & (global_x >= 0) & (global_x < inputCols);

        // Copy all the input of this block into shared memory.
        {
            inputPtr += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS;
            for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
            {
                fml value;
                if (!isInsideInput)
                    value = FML(0.0);
                else
                    value = inputPtr[inputComponentIndex];
                input_shared[threadIdx.y * (BLOCK_SIZE_X+KERNEL_COLS) + threadIdx.x + inputComponentIndex * (BLOCK_SIZE_Y*(BLOCK_SIZE_X+KERNEL_COLS)+INPUT_COALESCED_EXTRA_PADDING+KERNEL_ROWS*KERNEL_COLS)] = value;
            }
        }

        // Determine if this thread is (was?) an output thread or not.
        isOutputThread = (isInsideInput &
                          (threadIdx.x >= KERNEL_COLS/2) & (threadIdx.x < BLOCK_SIZE_X-KERNEL_COLS/2) &
                          (threadIdx.y >= KERNEL_ROWS/2) & (threadIdx.y < BLOCK_SIZE_Y-KERNEL_ROWS/2));
        dA_ptr += blockIdx.z * inputRows * inputCols * NUM_KERNELS + global_y * inputCols * NUM_KERNELS + global_x * NUM_KERNELS;
    }

    // Calculate, reduce, and store the value of db and of every dk.
    u32 linearThreadIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
    {
        // Copy our dA into shared memory.
        {
            fml dA;
            if (!isOutputThread)
                dA = FML(0.0);
            else
                dA = dA_ptr[kernelIndex];
            dA_shared[linearThreadIndex] = dA;
        }
        __syncthreads();

        // For every dk of this kernel, calculate it and store it.
        const fml* dA_start = dA_shared + (KERNEL_ROWS/2) * BLOCK_SIZE_X + KERNEL_COLS/2;
        u32 numThingsToCalculate = KERNEL_ROWS*KERNEL_COLS*INPUT_COMPONENTS;
        for (u32 thing = linearThreadIndex; thing < numThingsToCalculate; thing += BLOCK_SIZE_Y*BLOCK_SIZE_X)
        {
            u32 inputComponentIndex = thing / (KERNEL_ROWS*KERNEL_COLS);
            u32 kernelRowIndex = (thing % (KERNEL_ROWS*KERNEL_COLS)) / KERNEL_COLS;
            u32 kernelColIndex = (thing % (KERNEL_ROWS*KERNEL_COLS)) % KERNEL_COLS;
            const fml* input_start = input_shared + inputComponentIndex * (BLOCK_SIZE_Y*(BLOCK_SIZE_X+KERNEL_COLS)+INPUT_COALESCED_EXTRA_PADDING+KERNEL_ROWS*KERNEL_COLS) + kernelRowIndex * (BLOCK_SIZE_X+KERNEL_COLS) + kernelColIndex;
            fml dk = FML(0.0);
            for (u32 blockRowIndex = 0; blockRowIndex < BLOCK_SIZE_Y-KERNEL_ROWS+1; blockRowIndex++)
            {
                for (u32 blockColIndex = 0; blockColIndex < BLOCK_SIZE_X-KERNEL_COLS+1; blockColIndex++)
                {
                    fml input = input_start[blockRowIndex*(BLOCK_SIZE_X+KERNEL_COLS)+blockColIndex];
                    fml dA = dA_start[blockRowIndex*BLOCK_SIZE_X+blockColIndex];
                    dk += input * dA;
                }
            }
            fml* dk_here = dk_ptr + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS + kernelRowIndex * KERNEL_COLS * INPUT_COMPONENTS + kernelColIndex * INPUT_COMPONENTS + inputComponentIndex;
            atomicAdd(dk_here, dk * scaleFactor);
        }
        __syncthreads();

        // Reduce (via summation) the db values in this block.
        u32 size = BLOCK_SIZE_Y*BLOCK_SIZE_X;
        while (size > 1)
        {
            size /= 2;
            if (linearThreadIndex < size)
                dA_shared[linearThreadIndex] += dA_shared[linearThreadIndex + size];
            __syncthreads();
        }

        // Store the db value.
        if (linearThreadIndex == 0)
            atomicAdd(db_ptr + kernelIndex, dA_shared[0] * scaleFactor);
        __syncthreads();
    }
}


#if GPU_ACCUM_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_ACCUM_USE_TEMPLATE
accum_in_multiple_passes_templated(
#else
accum_in_multiple_passes(
#endif
#if !GPU_ACCUM_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,
              fml* dk_ptr,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    // We use shared memory so that each global memory value only must be read once!
    // Makes everything much much faster.
    // We will keep ONE components of the input block in shared memory at a time.
    // And we will also keep ONE component of the dA block in shared memory at a time.
    extern __shared__ fml memory_shared[];
    fml* input_shared = memory_shared + 0;  // <--v-- note the weird dimensions on input_shared; this is so that shared memory access is coalesced; be sure to index this array properly!
    fml* dA_shared    = input_shared + BLOCK_SIZE_Y*(BLOCK_SIZE_X+KERNEL_COLS);

    // Useful to have:
    bool isInsideInput;
    bool isOutputThread;
    {
        i32 global_y = blockIdx.y * (BLOCK_SIZE_Y-KERNEL_ROWS+1) + threadIdx.y;  global_y -= KERNEL_ROWS/2;
        i32 global_x = blockIdx.x * (BLOCK_SIZE_X-KERNEL_COLS+1) + threadIdx.x;  global_x -= KERNEL_COLS/2;

        // Determine if this thread can copy input pixels or not.
        isInsideInput = (global_y >= 0) & (global_y < inputRows) & (global_x >= 0) & (global_x < inputCols);
        inputPtr += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS;

        // Determine if this thread is (was?) an output thread or not.
        isOutputThread = (isInsideInput &
                          (threadIdx.x >= KERNEL_COLS/2) & (threadIdx.x < BLOCK_SIZE_X-KERNEL_COLS/2) &
                          (threadIdx.y >= KERNEL_ROWS/2) & (threadIdx.y < BLOCK_SIZE_Y-KERNEL_ROWS/2));
        dA_ptr += blockIdx.z * inputRows * inputCols * NUM_KERNELS + global_y * inputCols * NUM_KERNELS + global_x * NUM_KERNELS;
    }

    // Calculate, reduce, and store the value of db and of every dk.
    // (This will take several passes through the input, but only one pass through the dA...)
    u32 linearThreadIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
    {
        // Copy our dA into shared memory.
        {
            fml dA;
            if (!isOutputThread)
                dA = FML(0.0);
            else
                dA = dA_ptr[kernelIndex];
            dA_shared[linearThreadIndex] = dA;
        }

        // For every input channel...
        for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
        {
            // Copy this channel into shared memory.
            {
                fml input;
                if (!isInsideInput)
                    input = FML(0.0);
                else
                    input = inputPtr[inputComponentIndex];
                input_shared[threadIdx.y * (BLOCK_SIZE_X+KERNEL_COLS) + threadIdx.x] = input;
            }
            __syncthreads();

            // For every dk of this kernel (for this channel), calculate it and store it.
            const fml* dA_start = dA_shared + (KERNEL_ROWS/2) * BLOCK_SIZE_X + KERNEL_COLS/2;
            u32 numThingsToCalculate = KERNEL_ROWS*KERNEL_COLS;
            for (u32 thing = linearThreadIndex; thing < numThingsToCalculate; thing += BLOCK_SIZE_Y*BLOCK_SIZE_X)
            {
                u32 kernelRowIndex = thing / KERNEL_COLS;
                u32 kernelColIndex = thing % KERNEL_COLS;
                const fml* input_start = input_shared + kernelRowIndex * (BLOCK_SIZE_X+KERNEL_COLS) + kernelColIndex;
                fml dk = FML(0.0);
                for (u32 blockRowIndex = 0; blockRowIndex < BLOCK_SIZE_Y-KERNEL_ROWS+1; blockRowIndex++)
                {
                    for (u32 blockColIndex = 0; blockColIndex < BLOCK_SIZE_X-KERNEL_COLS+1; blockColIndex++)
                    {
                        fml input = input_start[blockRowIndex*(BLOCK_SIZE_X+KERNEL_COLS)+blockColIndex];
                        fml dA = dA_start[blockRowIndex*BLOCK_SIZE_X+blockColIndex];
                        dk += input * dA;
                    }
                }
                fml* dk_here = dk_ptr + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS + kernelRowIndex * KERNEL_COLS * INPUT_COMPONENTS + kernelColIndex * INPUT_COMPONENTS + inputComponentIndex;
                atomicAdd(dk_here, dk * scaleFactor);
            }
            __syncthreads();
        }

        // Reduce (via summation) the db values in this block.
        u32 size = BLOCK_SIZE_Y*BLOCK_SIZE_X;
        while (size > 1)
        {
            size /= 2;
            if (linearThreadIndex < size)
                dA_shared[linearThreadIndex] += dA_shared[linearThreadIndex + size];
            __syncthreads();
        }

        // Store the db value.
        if (linearThreadIndex == 0)
            atomicAdd(db_ptr + kernelIndex, dA_shared[0] * scaleFactor);
        __syncthreads();
    }
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
