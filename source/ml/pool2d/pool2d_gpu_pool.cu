#include <ml/pool2d/gpu.h>

#include "../cuda_stuff.ipp"


#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_X 32
#define DESIRED_BLOCKS_PER_SM 2


#define GPU_POOL2D_USE_TEMPLATE 0
#include "pool2d_gpu_pool_impl.ipp"
#undef  GPU_POOL2D_USE_TEMPLATE

#define GPU_POOL2D_USE_TEMPLATE 1
#include "pool2d_gpu_pool_impl.ipp"
#undef  GPU_POOL2D_USE_TEMPLATE


#define RUN_FALLBACK_IMPL \
    if (poolRows == 2 && poolCols == 2) \
        pool2d<2, 2><<<gridSize, blockSize>>>(inputComponents, inputPtr, inputRows, inputCols, outputPtr); \
    else \
        throw eImpossiblePath();


#define SWITCH_POOL_SIZE(inputComponents) \
    switch (poolRows * 0x10 + poolCols) \
    { \
        case 0x22: \
            pool2d_templated<inputComponents, 2, 2><<<gridSize, blockSize>>>(inputPtr, inputRows, inputCols, outputPtr); \
            break; \
        default: \
            throw eImpossiblePath(); \
    }


#define RUN_POOL_GPU_FUNCTION \
    switch (inputComponents) \
    { \
        case 1: SWITCH_POOL_SIZE(1)  break; \
        case 2: SWITCH_POOL_SIZE(2)  break; \
        case 3: SWITCH_POOL_SIZE(3)  break; \
        case 4: SWITCH_POOL_SIZE(4)  break; \
        case 5: SWITCH_POOL_SIZE(5)  break; \
        case 6: SWITCH_POOL_SIZE(6)  break; \
        case 7: SWITCH_POOL_SIZE(7)  break; \
        case 8: SWITCH_POOL_SIZE(8)  break; \
        case 16: SWITCH_POOL_SIZE(16)  break; \
        case 32: SWITCH_POOL_SIZE(32)  break; \
        case 64: SWITCH_POOL_SIZE(64)  break; \
        default: \
            RUN_FALLBACK_IMPL \
    }


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
    assert(inputCount > 0);
    assert(inputPtr);
    assert(inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(poolRows > 0 && poolCols > 0);
    assert(outputPtr);

    if (poolRows != 2 || poolCols != 2)
        throw eInvalidArgument("The poolRows and poolCols must each be 2 when using the GPU implementation.");

    dim3 gridSize;
    gridSize.x = (inputCols-1) / BLOCK_SIZE_X + 1;
    gridSize.y = (inputRows-1) / BLOCK_SIZE_Y + 1;
    gridSize.z = inputCount;

    dim3 blockSize;
    blockSize.x = BLOCK_SIZE_X;
    blockSize.y = BLOCK_SIZE_Y;
    blockSize.z = 1;

    RUN_POOL_GPU_FUNCTION

    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess)
        throw eRuntimeError(std::string("CUDA launch error: ") + cudaGetErrorString(errSync));
}


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
