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
    // TODO
}


void un_pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr)
{
    // TODO
}


void pool2d_multi_input_with_memcpy(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr)
{
    u32 outputRows = inputRows / poolRows;
    u32 outputCols = inputCols / poolCols;

    fml* inputPtr_gpu = s_cudaMalloc(inputRows*inputCols*inputComponents*inputCount);
    fml* outputPtr_gpu = s_cudaMalloc(outputRows*outputCols*inputComponents*inputCount);

    s_cudaCopyHostToDevice(inputPtr_gpu, inputPtr, inputRows*inputCols*inputComponents*inputCount);
    s_cudaCopyHostToDevice(outputPtr_gpu, outputPtr, outputRows*outputCols*inputComponents*inputCount);

    pool2d_multi_input(
        inputCount,
        inputPtr_gpu,  inputRows,   inputCols,   inputComponents,
                       poolRows,  poolCols,
        outputPtr_gpu
    );

    s_cudaCopyDeviceToHost(outputPtr, outputPtr_gpu, outputRows*outputCols*inputComponents*inputCount);

    s_cudaFree(inputPtr_gpu);
    s_cudaFree(outputPtr_gpu);
}


void un_pool2d_multi_input_with_memcpy(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr)
{
    // TODO
}


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
