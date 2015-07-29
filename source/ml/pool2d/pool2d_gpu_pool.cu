#include <ml/pool2d/gpu.h>

#include "../cuda_stuff.ipp"


namespace ml
{
namespace pool2d
{
namespace gpu
{


void pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr)
{
    assert(inputPtr);
    assert(inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(poolRows > 0 && poolCols > 0);
    assert(outputPtr);

    if (poolRows != 2 || poolCols != 2)
        throw eInvalidArgument("The poolRows and poolCols must each be 2.");

    dim3 gridSize;
    gridSize.x = (inputCols-1) / BLOCK_SIZE_X + 1;
    gridSize.y = (inputRows-1) / BLOCK_SIZE_Y + 1;
    gridSize.z = inputCount;

    dim3 blockSize;
    blockSize.x = BLOCK_SIZE_X;
    blockSize.y = BLOCK_SIZE_Y;
    blockSize.z = 1;

    // TODO

    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess)
        throw eRuntimeError(std::string("CUDA launch error: ") + cudaGetErrorString(errSync));
}


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
