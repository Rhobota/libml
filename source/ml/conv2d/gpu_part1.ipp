#define ENABLE_DEVICE_FUNCTIONS
#include "../common_nn.ipp"


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


/*
 * A block is filled by threads with access to shared memory.
 * This way we are using the most threads allowed: 1024
 */
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE_X 32


#if GPU_PART1_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, 1)
gpu_conv2d_multi_input(
#if !GPU_PART1_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,
        const fml* kernelPtr, u32 kernelStepY, u32 kernelStepX,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr, u32 outputRows, u32 outputCols)
{
    // We use shared memory so that each global memory value only must be read once!
    // Makes everything much much faster.
    // We will keep only one component of the input block in shared memory at a time.
    // We will keep all the components of all the kernels in shared memory at the same time though!
    extern __shared__ fml memory_shared[];
    fml* input_shared  = memory_shared + 0;
    fml* kernel_shared = input_shared  + BLOCK_SIZE_Y * BLOCK_SIZE_X;
    fml* bias_shared   = kernel_shared + KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;

    // Useful things to have:
    u32 effectiveBlockSizeY = BLOCK_SIZE_Y-KERNEL_ROWS+1;
    u32 effectiveBlockSizeX = BLOCK_SIZE_X-KERNEL_COLS+1;
    u32 block_offset_y = blockIdx.y * effectiveBlockSizeY;
    u32 block_offset_x = blockIdx.x * effectiveBlockSizeX;
    i32 global_y = block_offset_y + threadIdx.y;  global_y -= KERNEL_ROWS/2;
    i32 global_x = block_offset_x + threadIdx.x;  global_x -= KERNEL_COLS/2;
    u32 linearThreadIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;

    // All threads will help copy values into the shared memory. But not
    // all threads will be required to calculate output values. Only
    // threads that have all the following attributes will be required
    // to calculate output values:
    //   - be inside the effective block,
    //   - be inside the input, and
    //   - be aligned to the kernel step size.
    //
    // EDIT: THE ABOVE IS HOW WE USED TO DO THIS. READ BELOW.
    // Now what we do is shift down all the threads that should
    // be calculating output values so that they're all together at
    // the beginning of a thread block, which will group them better
    // into warps-which-all-calculate-things vs warps-which-do-not,
    // which will give less warp divergence, which will give us better
    // warp utilization.

    // Determine if this thread can copy input pixels or not.
    bool isInsideInput =
                (global_y >= 0 && global_y < inputRows &&
                 global_x >= 0 && global_x < inputCols);
    inputPtr += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS;

    // Determine if this thread is an output thread or not.
    bool isOutputThread;
    u32 input_start_shift;
    {
        u32 max_y = block_offset_y + effectiveBlockSizeY - 1;
        if (max_y >= inputRows)
            max_y = inputRows - 1;
        u32 min_y = ((block_offset_y + kernelStepY - 1) / kernelStepY) * kernelStepY;
        u32 num_outputs_y = (max_y - min_y) / kernelStepY + 1;  // This line will break if kernelStepY is too big (>=effectiveBlockSizeY?).

        u32 max_x = block_offset_x + effectiveBlockSizeX - 1;
        if (max_x >= inputCols)
            max_x = inputCols - 1;
        u32 min_x = ((block_offset_x + kernelStepX - 1) / kernelStepX) * kernelStepX;
        u32 num_outputs_x = (max_x - min_x) / kernelStepX + 1;  // This line will break if kernelStepX is too big (>=effectiveBlockSizeX?).

        u32 num_total_output = num_outputs_y * num_outputs_x;

        isOutputThread = linearThreadIndex < num_total_output;
        u32 centerRow = min_y + (linearThreadIndex / num_outputs_x) * kernelStepY;
        u32 centerCol = min_x + (linearThreadIndex % num_outputs_x) * kernelStepX;
        outputPtr += blockIdx.z * outputRows * outputCols * NUM_KERNELS + centerRow/kernelStepY * outputCols * NUM_KERNELS + centerCol/kernelStepX * NUM_KERNELS;
        input_start_shift = (centerRow-block_offset_y) * BLOCK_SIZE_X + (centerCol-block_offset_x);
    }

    // Copy all the kernels into shared memory.
    {
        u32 sizeToCopy = KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;
        for (u32 copyIndex = linearThreadIndex; copyIndex < sizeToCopy; copyIndex += BLOCK_SIZE_Y * BLOCK_SIZE_X)
        {
            kernel_shared[copyIndex] = kernelPtr[copyIndex];
        }
        sizeToCopy = NUM_KERNELS;
        for (u32 copyIndex = linearThreadIndex; copyIndex < sizeToCopy; copyIndex += BLOCK_SIZE_Y * BLOCK_SIZE_X)
        {
            bias_shared[copyIndex] = kernelBiases[copyIndex];
        }
    }

    // For each component of the input, we will process it independently.
    fml accumulators[NUM_KERNELS];
    for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
    {
        // Copy this channel into the shared memory.
        if (isInsideInput)
        {
            input_shared[linearThreadIndex] = inputPtr[inputComponentIndex];
        }
        else
        {
            input_shared[linearThreadIndex] = FML(0.0);
        }

        // Don't move on until all threads have copied the values they are each responsible for.
        // Because we are about to use all these values in a calculation.
        __syncthreads();

        // Do the convolution of this channel, and add it to the accumulator.
        // Not all threads have work here, because some threads exist only to copy the apron
        // values into shared memory, and some threads are not aligned to the kernel step size.
        if (isOutputThread)
        {
            const fml* input_start = input_shared + input_start_shift;

            for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
            {
                // The calculation.
                const fml* kernel_start = kernel_shared + kernelIndex * KERNEL_ROWS * KERNEL_COLS * INPUT_COMPONENTS + inputComponentIndex;
                fml result = FML(0.0);
                for (u32 kernelRowIndex = 0; kernelRowIndex < KERNEL_ROWS; kernelRowIndex++)
                {
                    const fml* kernel_row = kernel_start + kernelRowIndex * KERNEL_COLS * INPUT_COMPONENTS;
                    const fml* input_row = input_start + kernelRowIndex * BLOCK_SIZE_X;
                    for (u32 kernelColIndex = 0; kernelColIndex < KERNEL_COLS; kernelColIndex++)
                    {
                        fml k = kernel_row[kernelColIndex * INPUT_COMPONENTS];
                        fml i = input_row[kernelColIndex];
                        result += k * i;
                    }
                }

                // The storage to the accumulator (or output it if we're finished).
                if (inputComponentIndex == 0)
                {
                    if (INPUT_COMPONENTS == 1)
                        outputPtr[kernelIndex] = (result + bias_shared[kernelIndex]) * scaleFactor;
                    else
                        accumulators[kernelIndex] = result + bias_shared[kernelIndex];
                }
                else
                {
                    if (inputComponentIndex+1 < INPUT_COMPONENTS)
                        accumulators[kernelIndex] += result;
                    else
                        outputPtr[kernelIndex] = (accumulators[kernelIndex] + result) * scaleFactor;
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
