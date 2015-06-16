#include <ml/conv2d/gpu.h>

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


template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, 1)
gpu_conv2d_multi_input(
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
    fml* input_shared  = memory_shared;
    fml* kernel_shared = input_shared  + BLOCK_SIZE_Y * BLOCK_SIZE_X;
    fml* bias_shared   = kernel_shared + KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;

    // Useful things to have:
    inputPtr  += blockIdx.z * inputRows * inputCols * INPUT_COMPONENTS;
    outputPtr += blockIdx.z * outputRows * outputCols * NUM_KERNELS;
    i32 global_y = blockIdx.y * (BLOCK_SIZE_Y-KERNEL_ROWS+1) + threadIdx.y;  global_y -= KERNEL_ROWS/2;
    i32 global_x = blockIdx.x * (BLOCK_SIZE_X-KERNEL_COLS+1) + threadIdx.x;  global_x -= KERNEL_COLS/2;

    // All threads will help copy values into the shared memory. But not
    // all threads will be required to calculate output values. Only
    // threads that have all the following attributes will be required
    // to calculate output values:
    //   - be inside the effective block,
    //   - be inside the input, and
    //   - be aligned to the kernel step size.
    bool isInsideEffectiveBlock =
                (threadIdx.y >= KERNEL_ROWS/2 && (threadIdx.y - KERNEL_ROWS/2) < (BLOCK_SIZE_Y-KERNEL_ROWS+1) &&
                 threadIdx.x >= KERNEL_COLS/2 && (threadIdx.x - KERNEL_COLS/2) < (BLOCK_SIZE_X-KERNEL_COLS+1));
    bool isInsideInput =
                (global_y >= 0 && global_y < inputRows &&
                 global_x >= 0 && global_x < inputCols);
    bool isAlignedToKerenlStep =
                ((global_y % kernelStepY) == 0 &&
                 (global_x % kernelStepX) == 0);
    bool isOutputThread =
                (isInsideEffectiveBlock &&
                 isInsideInput &&
                 isAlignedToKerenlStep);

    // Copy all the kernels into shared memory.
    {
        u32 sizeToCopy = KERNEL_ROWS  * KERNEL_COLS   * INPUT_COMPONENTS * NUM_KERNELS;
        for (u32 copyIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x; copyIndex < sizeToCopy; copyIndex += BLOCK_SIZE_Y * BLOCK_SIZE_X)
        {
            kernel_shared[copyIndex] = kernelPtr[copyIndex];
        }
        sizeToCopy = NUM_KERNELS;
        for (u32 copyIndex = threadIdx.y * BLOCK_SIZE_X + threadIdx.x; copyIndex < sizeToCopy; copyIndex += BLOCK_SIZE_Y * BLOCK_SIZE_X)
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
            input_shared[threadIdx.y * BLOCK_SIZE_X + threadIdx.x] = *(inputPtr + global_y * inputCols * INPUT_COMPONENTS + global_x * INPUT_COMPONENTS + inputComponentIndex);
        }
        else
        {
            input_shared[threadIdx.y * BLOCK_SIZE_X + threadIdx.x] = FML(0.0);
        }

        // Don't move on until all threads have copied the values they are each responsible for.
        // Because we are about to use all these values in a calculation.
        __syncthreads();

        // Do the convolution of this channel, and add it to the accumulator.
        // Not all threads have work here, because some threads exist only to copy the apron
        // values into shared memory, and some threads are not aligned to the kernel step size.
        if (isOutputThread)
        {
            const fml* input_start = input_shared + (threadIdx.y - KERNEL_ROWS/2) * BLOCK_SIZE_X + threadIdx.x - KERNEL_COLS/2;

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
                        result += kernel_row[kernelColIndex * INPUT_COMPONENTS]
                                * input_row[kernelColIndex];
                    }
                }

                // The storage to the accumulator.
                if (inputComponentIndex == 0)
                {
                    accumulators[kernelIndex] = (result + bias_shared[kernelIndex]) * scaleFactor;
                }
                else
                {
                    accumulators[kernelIndex] += result * scaleFactor;
                }
            }
        }

        // Don't loop back up and start messing with shared memory again until all threads are finished
        // with the calculation above (which uses the current shared memory values).
        __syncthreads();
    }

    // Output the final results.
    if (isOutputThread)
    {
        outputPtr += global_y/kernelStepY * outputCols * NUM_KERNELS + global_x/kernelStepX * NUM_KERNELS;
        for (u32 kernelIndex = 0; kernelIndex < NUM_KERNELS; kernelIndex++)
        {
            outputPtr[kernelIndex] = accumulators[kernelIndex];
        }
    }
}


#define SWITCH_KERNEL_DIMS(inputComponents, numKernels) \
    switch ((kernelRows * 0x10) + kernelCols) \
    { \
        case 0x33: \
            gpu_conv2d_multi_input<inputComponents, 3, 3, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x35: \
            gpu_conv2d_multi_input<inputComponents, 3, 5, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x37: \
            gpu_conv2d_multi_input<inputComponents, 3, 7, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x53: \
            gpu_conv2d_multi_input<inputComponents, 5, 3, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x55: \
            gpu_conv2d_multi_input<inputComponents, 5, 5, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x57: \
            gpu_conv2d_multi_input<inputComponents, 5, 7, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x73: \
            gpu_conv2d_multi_input<inputComponents, 7, 3, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x75: \
            gpu_conv2d_multi_input<inputComponents, 7, 5, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x77: \
            gpu_conv2d_multi_input<inputComponents, 7, 7, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        default: \
            throw eImpossiblePath(); \
    } \


#define SWITCH_NUM_KERNELS(inputComponents) \
    switch (numKernels) \
    { \
        case 1: SWITCH_KERNEL_DIMS(inputComponents, 1) break; \
        case 2: SWITCH_KERNEL_DIMS(inputComponents, 2) break; \
        case 3: SWITCH_KERNEL_DIMS(inputComponents, 3) break; \
        case 4: SWITCH_KERNEL_DIMS(inputComponents, 4) break; \
        case 5: SWITCH_KERNEL_DIMS(inputComponents, 5) break; \
        case 6: SWITCH_KERNEL_DIMS(inputComponents, 6) break; \
        case 7: SWITCH_KERNEL_DIMS(inputComponents, 7) break; \
        case 8: SWITCH_KERNEL_DIMS(inputComponents, 8) break; \
        case 9: SWITCH_KERNEL_DIMS(inputComponents, 9) break; \
        case 10: SWITCH_KERNEL_DIMS(inputComponents, 10) break; \
        case 11: SWITCH_KERNEL_DIMS(inputComponents, 11) break; \
        case 12: SWITCH_KERNEL_DIMS(inputComponents, 12) break; \
        case 13: SWITCH_KERNEL_DIMS(inputComponents, 13) break; \
        case 14: SWITCH_KERNEL_DIMS(inputComponents, 14) break; \
        case 15: SWITCH_KERNEL_DIMS(inputComponents, 15) break; \
        case 16: SWITCH_KERNEL_DIMS(inputComponents, 16) break; \
        default: \
            throw eInvalidArgument("Unsupported numKernels"); \
    } \


void conv2d_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
    assert(inputPtr && inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(kernelPtr && (kernelRows % 2) == 1 && (kernelCols % 2) == 1);
    assert(kernelStepY > 0 && kernelStepX > 0 && numKernels > 0);
    assert(kernelBiases);
    assert(outputPtr);

    u32 kernelRadiusY = kernelRows / 2;
    u32 kernelRadiusX = kernelCols / 2;

    dim3 blockSize;
    blockSize.x = BLOCK_SIZE_X;
    blockSize.y = BLOCK_SIZE_Y;
    blockSize.z = 1;

    u32 effectiveBlockSizeY = BLOCK_SIZE_Y - 2*kernelRadiusY;  // Each block of threads will fill
    u32 effectiveBlockSizeX = BLOCK_SIZE_X - 2*kernelRadiusX;  // a smaller block of output, because we need
                                                               // an "apron" so that our kernel doesn't fall of
                                                               // the side and into no-where-land.

    dim3 gridSize;
    gridSize.x = (inputCols-1) / effectiveBlockSizeX + 1;
    gridSize.y = (inputRows-1) / effectiveBlockSizeY + 1;
    gridSize.z = inputCount;

    u32 outputRows = (inputRows - 1) / kernelStepY + 1;
    u32 outputCols = (inputCols - 1) / kernelStepX + 1;

    if (kernelRows != 3 && kernelRows != 5 && kernelRows != 7)
        throw eInvalidArgument("Unsupported kernelRows: must be 3, 5, or 7.");
    if (kernelCols != 3 && kernelCols != 5 && kernelCols != 7)
        throw eInvalidArgument("Unsupported kernelCols: must be 3, 5, or 7.");

    u32 sharedMemNeeded = (BLOCK_SIZE_Y * BLOCK_SIZE_X + kernelRows * kernelCols * inputComponents * numKernels + numKernels) * sizeof(fml);

    switch (inputComponents)
    {
        case 1: SWITCH_NUM_KERNELS(1) break;
//      case 2: SWITCH_NUM_KERNELS(2) break;
//      case 3: SWITCH_NUM_KERNELS(3) break;
//      case 4: SWITCH_NUM_KERNELS(4) break;
//      case 5: SWITCH_NUM_KERNELS(5) break;
        case 6: SWITCH_NUM_KERNELS(6) break;
//      case 7: SWITCH_NUM_KERNELS(7) break;
//      case 8: SWITCH_NUM_KERNELS(8) break;
//      case 9: SWITCH_NUM_KERNELS(9) break;
//      case 10: SWITCH_NUM_KERNELS(10) break;
//      case 11: SWITCH_NUM_KERNELS(11) break;
//      case 12: SWITCH_NUM_KERNELS(12) break;
//      case 13: SWITCH_NUM_KERNELS(13) break;
//      case 14: SWITCH_NUM_KERNELS(14) break;
//      case 15: SWITCH_NUM_KERNELS(15) break;
//      case 16: SWITCH_NUM_KERNELS(16) break;
        default:
            throw eInvalidArgument("Unsupported inputComponents");
    }

    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess)
        throw eRuntimeError(std::string("CUDA launch error: ") + cudaGetErrorString(errSync));
}


void conv2d_backprop_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
              fml* di_ptr,    u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
        const fml* dA_ptr)
{
    // TODO
}


void conv2d_accumError_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr, u32 inputRows,   u32 inputCols,   u32 inputComponents,
              fml* dk_ptr,   u32 kernelRows,  u32 kernelCols,
                             u32 kernelStepY, u32 kernelStepX,
                             u32 numKernels,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    // TODO
    // Don't forget to set dk_ptr and db_ptr vectors to zero before you begin.
}


void conv2d_multi_input_with_memcpy(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
    fml* inputPtr_gpu = s_cudaMalloc(inputCount * inputStride);
    fml* kernelPtr_gpu = s_cudaMalloc(kernelRows * kernelCols * inputComponents * numKernels);
    fml* kernelBiases_gpu = s_cudaMalloc(numKernels);
    fml* outputPtr_gpu = s_cudaMalloc(inputCount * outputStride);

    s_cudaCopyHostToDevice(inputPtr_gpu, inputPtr, inputCount * inputStride);
    s_cudaCopyHostToDevice(kernelPtr_gpu, kernelPtr, kernelRows * kernelCols * inputComponents * numKernels);
    s_cudaCopyHostToDevice(kernelBiases_gpu, kernelBiases, numKernels);

    conv2d_multi_input(
        inputCount,  inputStride,  outputStride,
        inputPtr_gpu,  inputRows,   inputCols,   inputComponents,
        kernelPtr_gpu, kernelRows,  kernelCols,
                              kernelStepY, kernelStepX,
                              numKernels,
        kernelBiases_gpu, scaleFactor,
        outputPtr_gpu
    );

    s_cudaCopyDeviceToHost(outputPtr, outputPtr_gpu, inputCount * outputStride);

    s_cudaFree(inputPtr_gpu);
    s_cudaFree(kernelPtr_gpu);
    s_cudaFree(kernelBiases_gpu);
    s_cudaFree(outputPtr_gpu);
}


void conv2d_backprop_multi_input_with_memcpy(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
              fml* di_ptr,    u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
        const fml* dA_ptr)
{
    fml* di_ptr_gpu = s_cudaMalloc(inputCount * inputStride);
    fml* kernelPtr_gpu = s_cudaMalloc(kernelRows * kernelCols * inputComponents * numKernels);
    fml* kernelBiases_gpu = s_cudaMalloc(numKernels);
    fml* dA_ptr_gpu = s_cudaMalloc(inputCount * outputStride);

    s_cudaCopyHostToDevice(kernelPtr_gpu, kernelPtr, kernelRows * kernelCols * inputComponents * numKernels);
    s_cudaCopyHostToDevice(kernelBiases_gpu, kernelBiases, numKernels);
    s_cudaCopyHostToDevice(dA_ptr_gpu, dA_ptr, inputCount * outputStride);

    conv2d_backprop_multi_input(
        inputCount,  inputStride,  outputStride,
        di_ptr_gpu,    inputRows,   inputCols,   inputComponents,
        kernelPtr_gpu, kernelRows,  kernelCols,
                              kernelStepY, kernelStepX,
                              numKernels,
        kernelBiases_gpu, scaleFactor,
        dA_ptr_gpu
    );

    s_cudaCopyDeviceToHost(di_ptr, di_ptr_gpu, inputCount * inputStride);

    s_cudaFree(di_ptr_gpu);
    s_cudaFree(kernelPtr_gpu);
    s_cudaFree(kernelBiases_gpu);
    s_cudaFree(dA_ptr_gpu);
}


void conv2d_accumError_multi_input_with_memcpy(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr, u32 inputRows,   u32 inputCols,   u32 inputComponents,
              fml* dk_ptr,   u32 kernelRows,  u32 kernelCols,
                             u32 kernelStepY, u32 kernelStepX,
                             u32 numKernels,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    fml* inputPtr_gpu = s_cudaMalloc(inputCount * inputStride);
    fml* dk_ptr_gpu = s_cudaMalloc(kernelRows * kernelCols * inputComponents * numKernels);
    fml* db_ptr_gpu = s_cudaMalloc(numKernels);
    fml* dA_ptr_gpu = s_cudaMalloc(inputCount * outputStride);

    s_cudaCopyHostToDevice(inputPtr_gpu, inputPtr, inputCount * inputStride);
    s_cudaCopyHostToDevice(dA_ptr_gpu, dA_ptr, inputCount * outputStride);

    conv2d_accumError_multi_input(
        inputCount,  inputStride,  outputStride,
        inputPtr_gpu, inputRows,   inputCols,   inputComponents,
        dk_ptr_gpu,   kernelRows,  kernelCols,
                             kernelStepY, kernelStepX,
                             numKernels,
        db_ptr_gpu, scaleFactor,
        dA_ptr_gpu
    );

    s_cudaCopyDeviceToHost(dk_ptr, dk_ptr_gpu, kernelRows * kernelCols * inputComponents * numKernels);
    s_cudaCopyDeviceToHost(db_ptr, db_ptr_gpu, numKernels);

    s_cudaFree(inputPtr_gpu);
    s_cudaFree(dk_ptr_gpu);
    s_cudaFree(db_ptr_gpu);
    s_cudaFree(dA_ptr_gpu);
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
