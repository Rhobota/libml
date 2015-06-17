#include <ml/conv2d/gpu.h>

#include "gpu_part1.ipp"


namespace ml
{
namespace conv2d
{
namespace gpu
{


#define SWITCH_KERNEL_DIMS(inputComponents, numKernels) \
    switch ((kernelRows * 0x10) + kernelCols) \
    { \
        case 0x33: \
            gpu_conv2d_multi_input<inputComponents, 3, 3, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x35: \
            gpu_conv2d_multi_input<inputComponents, 3, 5, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x37: \
            gpu_conv2d_multi_input<inputComponents, 3, 7, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x53: \
            gpu_conv2d_multi_input<inputComponents, 5, 3, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x55: \
            gpu_conv2d_multi_input<inputComponents, 5, 5, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x57: \
            gpu_conv2d_multi_input<inputComponents, 5, 7, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x73: \
            gpu_conv2d_multi_input<inputComponents, 7, 3, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x75: \
            gpu_conv2d_multi_input<inputComponents, 7, 5, numKernels><<<gridSize, blockSize>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, kernelStepY, kernelStepX, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
 \
        case 0x77: \
            gpu_conv2d_multi_input<inputComponents, 7, 7, numKernels><<<gridSize, blockSize>>>( \
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

    switch (inputComponents)
    {
        case 1: SWITCH_NUM_KERNELS(1) break;
        case 2: SWITCH_NUM_KERNELS(2) break;
        case 3: SWITCH_NUM_KERNELS(3) break;
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


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
