#include <ml/conv2d/gpu.h>

#define ENABLE_DEVICE_FUNCTIONS
#include "../common_nn.ipp"

#define GPU_PART1_USE_TEMPLATE 0
#include "gpu_part1.ipp"
#undef  GPU_PART1_USE_TEMPLATE

#define GPU_PART1_USE_TEMPLATE 1
#include "gpu_part1.ipp"
#undef  GPU_PART1_USE_TEMPLATE


/*
 * If defined, this file will take a long time to compile, but
 * will output very fast code. You should turn this on for
 * production code, but turn it off to do quick test iterations.
 */
#define COMPILE_A_BUNCH_OF_TEMPLATES_TO_MAKE_FAST_CODE 0


/*
 * If defined, this code will refuse to run the fallback
 * implementation, which is a very slow implementation.
 * This is nice to turn on if you want to be sure your
 * code uses the fast templated versions of the function
 * instead of the fallback.
 */
#define THROW_IF_FALLBACK_IMPL_NEEDED 0


namespace ml
{
namespace conv2d
{
namespace gpu
{


#if THROW_IF_FALLBACK_IMPL_NEEDED
#define RUN_FALLBACK_IMPL \
    throw eRuntimeError("The fallback (aka, slow) implementation is needed to convolve this input. But we've turned off the fallback implementation, so you'll need to turn it on or (preferably) modify your code to use one of the fast implementation paths.");
#else
#define RUN_FALLBACK_IMPL \
    gpu_conv2d_multi_input<<<gridSize, blockSize, sharedMemNeeded>>>( \
        inputComponents, kernelRows, kernelCols, kernelStepY, kernelStepX, numKernels, \
        inputPtr,  inputRows,   inputCols, \
        kernelPtr, \
        kernelBiases, scaleFactor, \
        outputPtr, outputRows, outputCols);
#endif


#define SWITCH_KERNEL_DIMS(inputComponents, kernelStepY, kernelStepX, numKernels) \
    switch ((kernelRows * 0x10) + kernelCols) \
    { \
        case 0x33: \
            gpu_conv2d_multi_input_templated<inputComponents, 3, 3, kernelStepY, kernelStepX, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
        case 0x55: \
            gpu_conv2d_multi_input_templated<inputComponents, 5, 5, kernelStepY, kernelStepX, numKernels><<<gridSize, blockSize, sharedMemNeeded>>>( \
                inputPtr,  inputRows,   inputCols, \
                kernelPtr, \
                kernelBiases, scaleFactor, \
                outputPtr, outputRows, outputCols); \
            break; \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#define SWITCH_KERNEL_STEP(inputComponents, numKernels) \
    switch ((kernelStepY * 0x10) + kernelStepX) \
    { \
        case 0x11: SWITCH_KERNEL_DIMS(inputComponents, 1, 1, numKernels)  break; \
        case 0x22: SWITCH_KERNEL_DIMS(inputComponents, 2, 2, numKernels)  break; \
        default: \
            RUN_FALLBACK_IMPL \
    } \


#define SWITCH_NUM_KERNELS(inputComponents) \
    switch (numKernels) \
    { \
        case 1: SWITCH_KERNEL_STEP(inputComponents, 1)  break; \
        case 2: SWITCH_KERNEL_STEP(inputComponents, 2)  break; \
        case 3: SWITCH_KERNEL_STEP(inputComponents, 3)  break; \
        case 4: SWITCH_KERNEL_STEP(inputComponents, 4)  break; \
        case 5: SWITCH_KERNEL_STEP(inputComponents, 5)  break; \
        case 6: SWITCH_KERNEL_STEP(inputComponents, 6)  break; \
        case 7: SWITCH_KERNEL_STEP(inputComponents, 7)  break; \
        case 8: SWITCH_KERNEL_STEP(inputComponents, 8)  break; \
        case 9: SWITCH_KERNEL_STEP(inputComponents, 9)  break; \
        case 10: SWITCH_KERNEL_STEP(inputComponents, 10)  break; \
        case 15: SWITCH_KERNEL_STEP(inputComponents, 15)  break; \
        case 16: SWITCH_KERNEL_STEP(inputComponents, 16)  break; \
        case 20: SWITCH_KERNEL_STEP(inputComponents, 20)  break; \
        case 25: SWITCH_KERNEL_STEP(inputComponents, 25)  break; \
        case 32: SWITCH_KERNEL_STEP(inputComponents, 32)  break; \
        case 50: SWITCH_KERNEL_STEP(inputComponents, 50)  break; \
        case 64: SWITCH_KERNEL_STEP(inputComponents, 64)  break; \
        case 100: SWITCH_KERNEL_STEP(inputComponents, 100)  break; \
        default: \
            RUN_FALLBACK_IMPL \
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

    if (kernelRows != 1 && kernelRows != 3 && kernelRows != 5 && kernelRows != 7)
        throw eInvalidArgument("Unsupported kernelRows: must be 1, 3, 5, or 7.");
    if (kernelCols != 1 && kernelCols != 3 && kernelCols != 5 && kernelCols != 7)
        throw eInvalidArgument("Unsupported kernelCols: must be 1, 3, 5, or 7.");

    if (kernelStepY > 5)
        throw eInvalidArgument("Unsupported kernelStepY: must be in [1,5].");
    if (kernelStepX > 5)
        throw eInvalidArgument("Unsupported kernelStepX: must be in [1,5].");

    if (numKernels > MAX_KERNELS_SUPPORTED)
        throw eInvalidArgument("Unsupported numKernels: you specified too many!");

    u32 kernelRadiusY = kernelRows / 2;
    u32 kernelRadiusX = kernelCols / 2;

    u32 effectiveBlockSizeY = BLOCK_SIZE_Y - 2*kernelRadiusY;  // Each block of threads will fill
    u32 effectiveBlockSizeX = BLOCK_SIZE_X - 2*kernelRadiusX;  // a smaller block of output, because we need
                                                               // an "apron" so that our kernel doesn't fall of
                                                               // the side and into no-where-land.

    dim3 gridSize;
    gridSize.x = (inputCols-1) / effectiveBlockSizeX + 1;
    gridSize.y = (inputRows-1) / effectiveBlockSizeY + 1;
    gridSize.z = inputCount;

    dim3 blockSize;
    blockSize.x = BLOCK_SIZE_X;
    blockSize.y = BLOCK_SIZE_Y;
    blockSize.z = 1;

    u32 sharedMemNeeded = (BLOCK_SIZE_Y*BLOCK_SIZE_X + kernelRows*kernelCols*inputComponents*numKernels + numKernels) * sizeof(fml);

    u32 outputRows = (inputRows - 1) / kernelStepY + 1;
    u32 outputCols = (inputCols - 1) / kernelStepX + 1;

#if COMPILE_A_BUNCH_OF_TEMPLATES_TO_MAKE_FAST_CODE
    switch (inputComponents)
    {
        case 1: SWITCH_NUM_KERNELS(1)  break;
        case 2: SWITCH_NUM_KERNELS(2)  break;
        case 3: SWITCH_NUM_KERNELS(3)  break;
        case 4: SWITCH_NUM_KERNELS(4)  break;
        case 5: SWITCH_NUM_KERNELS(5)  break;
        case 6: SWITCH_NUM_KERNELS(6)  break;
        case 7: SWITCH_NUM_KERNELS(7)  break;
        case 8: SWITCH_NUM_KERNELS(8)  break;
        case 9: SWITCH_NUM_KERNELS(9)  break;
        case 10: SWITCH_NUM_KERNELS(10)  break;
        case 15: SWITCH_NUM_KERNELS(15)  break;
        case 16: SWITCH_NUM_KERNELS(16)  break;
        case 20: SWITCH_NUM_KERNELS(20)  break;
        case 25: SWITCH_NUM_KERNELS(25)  break;
        case 32: SWITCH_NUM_KERNELS(32)  break;
        case 50: SWITCH_NUM_KERNELS(50)  break;
        case 64: SWITCH_NUM_KERNELS(64)  break;
        case 100: SWITCH_NUM_KERNELS(100)  break;
        default:
            RUN_FALLBACK_IMPL
    }
#else
    RUN_FALLBACK_IMPL
#endif

    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess)
        throw eRuntimeError(std::string("CUDA launch error: ") + cudaGetErrorString(errSync));
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
