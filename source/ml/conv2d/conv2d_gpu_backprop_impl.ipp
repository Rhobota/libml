/*
 * Note: CNNs has a concept of "kernel" and CUDA has a concept of "kernel". That could
 * cause confusion. In this file we will only talk about CNN kernels, and we will avoid
 * using the term "kernel" to talk about that CUDA concept--we will use alternate words
 * for that instead.
 */


namespace ml
{
namespace conv2d
{
namespace gpu
{


#if GPU_BACKPROP_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_BACKPROP_USE_TEMPLATE
backprop_in_one_pass_templated(
#else
backprop_in_one_pass(
#endif
#if !GPU_BACKPROP_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
              fml* di_ptr,  u32 inputRows,   u32 inputCols,
        const fml* kernelPtr,
        fml scaleFactor,
        const fml* dA_ptr)
{
    // We use shared memory so that each global memory value only must be read once!
    // Makes everything much much faster.
    // We will keep all component of the input block in shared memory at the same time.
    // And we will keep all the components of all the kernels in shared memory at the same time!
    extern __shared__ fml memory_shared[];
    fml* kernel_shared = memory_shared + 0;
    fml* input_shared  = kernel_shared + KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;

    // Copy all the kernels into shared memory.
    {
        u32 sizeToCopy = KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;
        for (u32 copyIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x; copyIndex < sizeToCopy; copyIndex += BLOCK_SIZE_Y * BLOCK_SIZE_X)
        {
            kernel_shared[copyIndex] = kernelPtr[copyIndex];
        }
    }

    bool isOutputThread;
    u32 input_start_offset;
    {
        i32 global_y = blockIdx.y * (BLOCK_SIZE_Y-KERNEL_ROWS+1) + threadIdx.y;  global_y -= KERNEL_ROWS/2;
        i32 global_x = blockIdx.x * (BLOCK_SIZE_X-KERNEL_COLS+1) + threadIdx.x;  global_x -= KERNEL_COLS/2;
        bool isInsideInput = (global_y >= 0) & (global_y < inputRows) & (global_x >= 0) & (global_x < inputCols);

        // Copy all the input of this block into shared memory.
        {
            u32 linearThreadIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
            dA_ptr += blockIdx.z * inputRows * inputCols * NUM_KERNELS + global_y * inputCols * NUM_KERNELS + global_x * NUM_KERNELS;
            for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
            {
                fml value;
                if (!isInsideInput)
                    value = FML(0.0);
                else
                    value = dA_ptr[kernelIndex];
                input_shared[linearThreadIndex + kernelIndex * BLOCK_SIZE_Y * BLOCK_SIZE_X] = value;
            }
        }

        // Determine if this thread is an output thread or not.
        {
            // All threads will help copy values into the shared memory. But not
            // all threads will be required to calculate output values. Only
            // threads that have all the following attributes will be required
            // to calculate output values:
            //   - be inside the effective block, and
            //   - be inside the input.
            isOutputThread = (isInsideInput &
                              (threadIdx.x >= KERNEL_COLS/2) & (threadIdx.x < BLOCK_SIZE_X-KERNEL_COLS/2) &
                              (threadIdx.y >= KERNEL_ROWS/2) & (threadIdx.y < BLOCK_SIZE_Y-KERNEL_ROWS/2));
            di_ptr += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS;
            input_start_offset = (threadIdx.y - KERNEL_ROWS/2) * BLOCK_SIZE_X + (threadIdx.x - KERNEL_COLS/2);
        }
    }

    // Don't move on until all threads have copied the values they are each responsible for.
    // Because we are about to use all these values in a calculation.
    __syncthreads();

    // Do the convolution.
    // Not all threads have work here, because some threads exist only to copy the apron
    // values into shared memory.
    if (isOutputThread)
    {
        for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
        {
            // The calculation.
            fml result = FML(0.0);
            for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
            {
                const fml* kernel_start = kernel_shared + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS + inputComponentIndex;
                const fml* input_start = input_shared + input_start_offset + kernelIndex * BLOCK_SIZE_Y * BLOCK_SIZE_X;
                for (u32 kernelRowIndex = 0; kernelRowIndex < KERNEL_ROWS; kernelRowIndex++)
                {
                    const fml* kernel_row = kernel_start + (KERNEL_ROWS-kernelRowIndex-1) * KERNEL_COLS * INPUT_COMPONENTS;
                    const fml* input_row = input_start + kernelRowIndex * BLOCK_SIZE_X;
                    for (u32 kernelColIndex = 0; kernelColIndex < KERNEL_COLS; kernelColIndex++)
                    {
                        fml k = kernel_row[(KERNEL_COLS-kernelColIndex-1) * INPUT_COMPONENTS];
                        fml i = input_row[kernelColIndex];
                        result += k * i;
                    }
                }
            }

            // The storage.
            di_ptr[inputComponentIndex] = result * scaleFactor;
        }
    }
}


#if GPU_BACKPROP_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_BACKPROP_USE_TEMPLATE
backprop_in_multiple_passes_templated(
#else
backprop_in_multiple_passes(
#endif
#if !GPU_BACKPROP_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
              fml* di_ptr,  u32 inputRows,   u32 inputCols,
        const fml* kernelPtr,
        fml scaleFactor,
        const fml* dA_ptr)
{
    // We use shared memory so that each global memory value only must be read once!
    // Makes everything much much faster.
    // We will keep only one component of the input block in shared memory at a time.
    // We will keep all the components of all the kernels in shared memory at the same time though!
    extern __shared__ fml memory_shared[];
    fml* kernel_shared = memory_shared + 0;
    fml* input_shared  = kernel_shared + KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;

    // Copy all the kernels into shared memory.
    {
        u32 sizeToCopy = KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;
        for (u32 copyIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x; copyIndex < sizeToCopy; copyIndex += BLOCK_SIZE_Y * BLOCK_SIZE_X)
        {
            kernel_shared[copyIndex] = kernelPtr[copyIndex];
        }
    }

    // Useful things to have:
    bool isInsideInput;
    bool isOutputThread;
    const fml* input_start;
    {
        i32 global_y = (blockIdx.y * (BLOCK_SIZE_Y-KERNEL_ROWS+1)) + threadIdx.y;  global_y -= KERNEL_ROWS/2;
        i32 global_x = (blockIdx.x * (BLOCK_SIZE_X-KERNEL_COLS+1)) + threadIdx.x;  global_x -= KERNEL_COLS/2;

        // Determine if this thread can copy input pixels or not.
        isInsideInput =
                    ((global_y >= 0) & (global_y < inputRows) &
                     (global_x >= 0) & (global_x < inputCols));
        dA_ptr += blockIdx.z * inputRows * inputCols * NUM_KERNELS + global_y * inputCols * NUM_KERNELS + global_x * NUM_KERNELS;

        // Determine if this thread is an output thread or not.
        //
        // All threads will help copy values into the shared memory. But not
        // all threads will be required to calculate output values. Only
        // threads that have all the following attributes will be required
        // to calculate output values:
        //   - be inside the effective block, and
        //   - be inside the input.
        isOutputThread = (isInsideInput &
                          (threadIdx.x >= KERNEL_COLS/2) & (threadIdx.x < BLOCK_SIZE_X-KERNEL_COLS/2) &
                          (threadIdx.y >= KERNEL_ROWS/2) & (threadIdx.y < BLOCK_SIZE_Y-KERNEL_ROWS/2));
        di_ptr += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS;
        input_start = input_shared + (threadIdx.y - KERNEL_ROWS/2) * BLOCK_SIZE_X + (threadIdx.x - KERNEL_COLS/2);
    }

    // For each component of the input, we will process it independently.
#if GPU_BACKPROP_USE_TEMPLATE
    fml accumulators[INPUT_COMPONENTS];
#else
    fml accumulators[MAX_INPUT_COMPONENTS_SUPPORTED];
#endif
    //#pragma unroll
    for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
    {
        // Copy this channel into the shared memory.
        {
            fml value;
            if (isInsideInput)
                value = dA_ptr[kernelIndex];
            else
                value = FML(0.0);
            input_shared[threadIdx.y * BLOCK_SIZE_X + threadIdx.x] = value;
        }

        // Don't move on until all threads have copied the values they are each responsible for.
        // Because we are about to use all these values in a calculation.
        __syncthreads();

        // Do the convolution of this channel, and add it to the accumulator.
        // Not all threads have work here, because some threads exist only to copy the apron
        // values into shared memory.
        if (isOutputThread)
        {
            for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
            {
                // The calculation.
                fml result = FML(0.0);
                const fml* kernel_start = kernel_shared + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS + inputComponentIndex;
                for (u32 kernelRowIndex = 0; kernelRowIndex < KERNEL_ROWS; kernelRowIndex++)
                {
                    const fml* kernel_row = kernel_start + (KERNEL_ROWS-kernelRowIndex-1) * KERNEL_COLS * INPUT_COMPONENTS;
                    const fml* input_row = input_start + kernelRowIndex * BLOCK_SIZE_X;
                    for (u32 kernelColIndex = 0; kernelColIndex < KERNEL_COLS; kernelColIndex++)
                    {
                        fml k = kernel_row[(KERNEL_COLS-kernelColIndex-1) * INPUT_COMPONENTS];
                        fml i = input_row[kernelColIndex];
                        result += k * i;
                    }
                }

                // The storage to the accumulator (or output it if we're finished).
                if (kernelIndex == 0)
                {
                    if (NUM_KERNELS == 1)
                        di_ptr[inputComponentIndex] = result * scaleFactor;
                    else
                        accumulators[inputComponentIndex] = result;
                }
                else
                {
                    if (kernelIndex+1 < NUM_KERNELS)
                        accumulators[inputComponentIndex] += result;
                    else
                        di_ptr[inputComponentIndex] = (accumulators[inputComponentIndex] + result) * scaleFactor;
                }
            }
        }

        // Don't loop back up and start messing with shared memory again until all threads are finished
        // with the calculation above (which uses the current shared memory values).
        __syncthreads();
    }
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
