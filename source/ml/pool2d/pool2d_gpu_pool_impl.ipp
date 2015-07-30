namespace ml
{
namespace pool2d
{
namespace gpu
{


#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(!!(COND))*2-1]
#define COMPILE_TIME_ASSERT3(X,L) STATIC_ASSERT(X,at_line_##L)
#define COMPILE_TIME_ASSERT2(X,L) COMPILE_TIME_ASSERT3(X,L)
#define COMPILE_TIME_ASSERT(X)    COMPILE_TIME_ASSERT2(X,__LINE__)


#if GPU_POOL2D_USE_TEMPLATE
template <u32 INPUT_COMPONENTS, u32 POOL_ROWS, u32 POOL_COLS>
#else
template <u32 POOL_ROWS, u32 POOL_COLS>
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
        u32 INPUT_COMPONENTS,
#endif
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,
              fml* outputPtr)
{
    COMPILE_TIME_ASSERT((BLOCK_SIZE_Y % POOL_ROWS) == 0);
    COMPILE_TIME_ASSERT((BLOCK_SIZE_X % POOL_COLS) == 0);

    __shared__ fml input_shared[BLOCK_SIZE_Y*BLOCK_SIZE_X];

    u32 global_y = blockIdx.y*BLOCK_SIZE_Y + threadIdx.y;
    u32 global_x = blockIdx.x*BLOCK_SIZE_X + threadIdx.x;
    bool isInsideInput = (global_y < inputRows) & (global_x < inputCols);
    bool isOutputThread =
            ((global_y % POOL_ROWS) == 0)         & ((global_x % POOL_COLS) == 0)          &
            ((global_y + POOL_ROWS) <= inputRows) & ((global_x + POOL_COLS) <= inputCols);
    inputPtr += blockIdx.z*inputRows*inputCols*INPUT_COMPONENTS + global_y*inputCols*INPUT_COMPONENTS + global_x*INPUT_COMPONENTS;
    outputPtr += blockIdx.z*(inputRows/POOL_ROWS)*(inputCols/POOL_COLS)*INPUT_COMPONENTS + (global_y/POOL_ROWS)*(inputCols/POOL_COLS)*INPUT_COMPONENTS + (global_x/POOL_COLS)*INPUT_COMPONENTS;

    for (u32 inputComponentIndex = 0; inputComponentIndex < INPUT_COMPONENTS; inputComponentIndex++)
    {
        fml value;
        if (!isInsideInput)
            value = FML(0.0);
        else
            value = inputPtr[inputComponentIndex];
        input_shared[threadIdx.y * BLOCK_SIZE_X + threadIdx.x] = value;

        __syncthreads();

        if (isOutputThread)
        {
            for (u32 r = 0; r < POOL_ROWS; r++)
            {
                for (u32 c = 0; c < POOL_COLS; c++)
                {
                    value = FML(fmax)(value, input_shared[(threadIdx.y+r) * BLOCK_SIZE_X + (threadIdx.x+c)]);
                }
            }

            outputPtr[inputComponentIndex] = value;
        }

        __syncthreads();
    }
}


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
