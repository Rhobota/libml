#include <ml/conv2d/gpu.h>

#include "gpu_conv.ipp"


namespace ml
{
namespace conv2d
{
namespace gpu
{


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

    bool canUseFastImpl = true;
    u32 sharedMemNeeded = (BLOCK_SIZE_Y*BLOCK_SIZE_X*inputComponents + kernelRows*kernelCols*inputComponents*numKernels + numKernels) * sizeof(fml);
    if (sharedMemNeeded * DESIRED_BLOCKS_PER_SM > SHARED_MEM_AVAIL_PER_SM)
    {
        canUseFastImpl = false;
        sharedMemNeeded = (BLOCK_SIZE_Y*BLOCK_SIZE_X + kernelRows*kernelCols*inputComponents*numKernels + numKernels) * sizeof(fml);
    }

    u32 outputRows = (inputRows - 1) / kernelStepY + 1;
    u32 outputCols = (inputCols - 1) / kernelStepX + 1;

    RUN_CONV2D_GPU_FUNTION

    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess)
        throw eRuntimeError(std::string("CUDA launch error: ") + cudaGetErrorString(errSync));
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml