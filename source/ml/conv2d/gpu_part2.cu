#include <ml/conv2d/gpu.h>

//#include "gpu_part2.ipp"


namespace ml
{
namespace conv2d
{
namespace gpu
{


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


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
