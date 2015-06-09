namespace ml2
{
namespace    // <-- un-named namespaces act like everything inside is statically scoped
{


void s_conv2d_multi_input(
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,  u32 inputStride,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr, u32 outputStride)
{
    // TODO
}


}  // end of anonymous namespace
}  // end of namespace ml2
