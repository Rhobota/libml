/*
 * Note: CNNs has a concept of "kernel" and CUDA has a concept of "kernel". That could
 * cause confusion. In this file we will only talk about CNN kernels, and we will avoid
 * using the term "kernel" to talk about that CUDA concept--we will use alternate words
 * for that instead.
 */


namespace ml
{
namespace conv2d
{
namespace gpu
{


#if GPU_ACCUM_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_ACCUM_USE_TEMPLATE
accum_in_one_pass_templated(
#else
accum_in_one_pass(
#endif
#if !GPU_ACCUM_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,
              fml* dk_ptr,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    // TODO
}


#if GPU_ACCUM_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_ACCUM_USE_TEMPLATE
accum_in_multiple_passes_templated(
#else
accum_in_multiple_passes(
#endif
#if !GPU_ACCUM_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 KERNEL_ROWS, u32 KERNEL_COLS, u32 NUM_KERNELS,
#endif
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,
              fml* dk_ptr,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    // TODO
}


}  // namespace gpu
}  // namespace conv2d
}  // namespace ml
