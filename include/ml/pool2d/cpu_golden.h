#include <ml/common.h>


/*
 * This file has the gold standard implementation of these functions.
 * Any new implementations should be tested against these.
 */


namespace ml
{
namespace pool2d
{
namespace cpu_golden
{


void pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr);


void un_pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr);


}  // namespace cpu_golden
}  // namespace pool2d
}  // namespace ml
