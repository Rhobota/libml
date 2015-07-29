#include <ml/pool2d/gpu.h>

#include "../cuda_stuff.ipp"


namespace ml
{
namespace pool2d
{
namespace gpu
{


void un_pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr)
{
    assert(inputPtr);
    assert(inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(poolRows > 0 && poolCols > 0);
    assert(srcPtr);
    assert(destPtr);

    // TODO
}


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
