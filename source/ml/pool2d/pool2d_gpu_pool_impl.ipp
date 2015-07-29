namespace ml
{
namespace pool2d
{
namespace gpu
{


#if GPU_POOL2D_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 POOL_ROWS, u32 POOL_COLS>
#endif
__global__
void
__launch_bounds__(BLOCK_SIZE_Y*BLOCK_SIZE_X, DESIRED_BLOCKS_PER_SM)
#if GPU_POOL2D_USE_TEMPLATE
pool2d_templated(
#else
pool2d(
#endif
#if !GPU_POOL2D_USE_TEMPLATE
        u32 INPUT_COMPONENTS, u32 POOL_ROWS, u32 POOL_COLS,
#endif
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,
              fml* outputPtr)
{
    // TODO
}


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
