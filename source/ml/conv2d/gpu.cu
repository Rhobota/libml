#include <ml/conv2d/gpu.h>

#define ENABLE_DEVICE_FUNCTIONS
#include "../common_nn.ipp"


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
    // TODO
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


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
