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
