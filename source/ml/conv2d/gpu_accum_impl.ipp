/*
 * Note: CNNs has a concept of "kernel" and CUDA has a concept of "kernel". That could
 * cause confusion. In this file we will only talk about CNN kernels, and we will avoid
 * using the term "kernel" to talk about that CUDA concept--we will use alternate words
 * for that instead.
 */

#include "../Cub/cub/cub.cuh"


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
    // We will keep all component of the input block in shared memory at the same time.
    // And we will also allocate some shared memory that is needed by cub::BlockReduce.
    extern __shared__ fml memory_shared[];
    fml* input_shared  = memory_shared + 0;
    typedef cub::BlockReduce<fml, BLOCK_SIZE_X, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BLOCK_SIZE_Y> BlockReduce;  // Also try: BLOCK_REDUCE_WARP_REDUCTIONS
    __shared__ typename BlockReduce::TempStorage blockreduce_temp_storage;

    // Useful to have:
    u32 linearThreadIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    bool isOutputThread;
    u32 input_start_offset;
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
                input_shared[linearThreadIndex + inputComponentIndex * BLOCK_SIZE_Y * BLOCK_SIZE_X] = value;
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
            dA_ptr += blockIdx.z * inputRows * inputCols * NUM_KERNELS + global_y * inputCols * NUM_KERNELS + global_x * NUM_KERNELS;
            input_start_offset = (threadIdx.y - KERNEL_ROWS/2) * BLOCK_SIZE_X + (threadIdx.x - KERNEL_COLS/2);
        }
    }

    // Don't move on until all threads have copied the values they are each responsible for.
    // Because we are about to use all these values in a calculation.
    __syncthreads();

    // Calculate, reduce, and store the value of db and of every dk.
    for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
    {
        // Grab the dA value here.
        fml dA;
        if (!isOutputThread)
            dA = FML(0.0);
        else
            dA = dA_ptr[kernelIndex];

        // Reduce (via summation) the db values in this block.
        fml db = BlockReduce(blockreduce_temp_storage).Sum(dA);
        __syncthreads();

        // Store the db value.
        if (linearThreadIndex == 0)
            atomicAdd(db_ptr + kernelIndex, db * scaleFactor);

        // For every dk:
        fml* dk_start = dk_ptr + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS;
        for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
        {
            const fml* input_start = input_shared + input_start_offset + inputComponentIndex * BLOCK_SIZE_Y * BLOCK_SIZE_X;
            for (u32 kernelRowIndex = 0; kernelRowIndex < KERNEL_ROWS; kernelRowIndex++)
            {
                fml* dk_row = dk_start + kernelRowIndex * KERNEL_COLS * INPUT_COMPONENTS;
                const fml* input_row = input_start + kernelRowIndex * BLOCK_SIZE_X;
                for (u32 kernelColIndex = 0; kernelColIndex < KERNEL_COLS; kernelColIndex++)
                {
                    // Calcuate the dk at this spot.
                    fml input;
                    if (!isOutputThread)
                        input = FML(0.0);
                    else
                        input = input_row[kernelColIndex];
                    fml dk = dA * input;

                    // Reduce (via summation) the dk values in this block.
                    dk = BlockReduce(blockreduce_temp_storage).Sum(dk);
                    __syncthreads();

                    // Store the dk value.
                    if (linearThreadIndex == 0)
                        atomicAdd(dk_row + kernelColIndex * INPUT_COMPONENTS + inputComponentIndex, dk * scaleFactor);
                }
            }
        }
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
    // We will keep only one component of the input block in shared memory at a time.
    // And we will also allocate some shared memory that is needed by cub::BlockReduce.
    extern __shared__ fml memory_shared[];
    fml* input_shared  = memory_shared + 0;
    typedef cub::BlockReduce<fml, BLOCK_SIZE_X, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BLOCK_SIZE_Y> BlockReduce;  // Also try: BLOCK_REDUCE_WARP_REDUCTIONS
    __shared__ typename BlockReduce::TempStorage blockreduce_temp_storage;

    // Useful things to have:
    u32 linearThreadIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
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
        inputPtr += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS;

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
        dA_ptr += blockIdx.z * inputRows * inputCols * NUM_KERNELS + global_y * inputCols * NUM_KERNELS + global_x * NUM_KERNELS;
        input_start = input_shared + (threadIdx.y - KERNEL_ROWS/2) * BLOCK_SIZE_X + (threadIdx.x - KERNEL_COLS/2);
    }

    // Store the dA values we'll need, so we can use them over and over later.
#if GPU_ACCUM_USE_TEMPLATE
    fml dA_local[NUM_KERNELS];
#else
    fml dA_local[MAX_KERNELS_SUPPORTED];
#endif
    for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
    {
        // Grab the dA value here.
        fml dA;
        if (!isOutputThread)
            dA = FML(0.0);
        else
            dA = dA_ptr[kernelIndex];
        dA_local[kernelIndex] = dA;

        // Reduce (via summation) the db values in this block.
        fml db = BlockReduce(blockreduce_temp_storage).Sum(dA);
        __syncthreads();

        // Store the db value.
        if (linearThreadIndex == 0)
            atomicAdd(db_ptr + kernelIndex, db * scaleFactor);
    }

    // For each component of the input, we will process it independently.
    for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
    {
        // Copy this channel (aka, component) into the shared memory.
        {
            fml value;
            if (!isInsideInput)
                value = FML(0.0);
            else
                value = inputPtr[inputComponentIndex];
            input_shared[linearThreadIndex] = value;
        }

        // Don't move on until all threads have copied the values they are each responsible for.
        // Because we are about to use all these values in a calculation.
        __syncthreads();

        // Calculate, reduce, and store the value of every dk.
        for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
        {
            // Grab the dA value here.
            fml dA = dA_local[kernelIndex];

            // For every dk:
            fml* dk_start = dk_ptr + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS + inputComponentIndex;
            for (u32 kernelRowIndex = 0; kernelRowIndex < KERNEL_ROWS; kernelRowIndex++)
            {
                const fml* input_row = input_start + kernelRowIndex * BLOCK_SIZE_X;
                fml* dk_row = dk_start + kernelRowIndex * KERNEL_COLS * INPUT_COMPONENTS;
                for (u32 kernelColIndex = 0; kernelColIndex < KERNEL_COLS; kernelColIndex++)
                {
                    // Calcuate the dk at this spot.
                    fml input;
                    if (!isOutputThread)
                        input = FML(0.0);
                    else
                        input = input_row[kernelColIndex];
                    fml dk = dA * input;

                    // Reduce (via summation) the dk values in this block.
                    dk = BlockReduce(blockreduce_temp_storage).Sum(dk);
                    __syncthreads();

                    // Store the dk value.
                    if (linearThreadIndex == 0)
                        atomicAdd(dk_row + kernelColIndex * INPUT_COMPONENTS, dk * scaleFactor);
                }
            }
        }
    }
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
