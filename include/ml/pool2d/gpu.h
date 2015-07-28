#include <ml/common.h>


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
              fml* outputPtr);


void un_pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr);


/*
 * The following functions are just helpers to test this GPU implementation.
 * They assume you're passing in host buffers, so these functions will allocate
 * device buffers and will copy the input to them. Also, these functions will
 * copy the output back into the host buffers. Don't use these in real production
 * code because obviously all that copying is not efficient! Use the functions
 * ABOVE instead because those assume you're passing device pointers so no copying
 * is needed.
 */

void pool2d_multi_input_with_memcpy(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr);


void un_pool2d_multi_input_with_memcpy(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr);


}  // namespace gpu
}  // namespace pool2d
}  // namespace ml
