#include <ml/pool2d/cpu_golden.h>

#include "../Eigen.h"


namespace ml
{
namespace pool2d
{
namespace cpu_golden
{


static
void s_pool2d(
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr)
{
    u32 outRows = inputRows / poolRows;
    u32 outCols = inputCols / poolCols;

    for (u32 r = 0; r < outRows; r++)
    {
        for (u32 c = 0; c < outCols; c++)
        {
            for (u32 i = 0; i < inputComponents; i++)
            {
                MapRowMajorWithStrideConst input(inputPtr+i,
                                                 inputRows, inputCols,
                                                 Stride(inputCols*inputComponents, inputComponents));
                fml maxValue = input.block(r*poolRows, c*poolCols, poolRows, poolCols).maxCoeff();
                *outputPtr++ = maxValue;
            }
        }
    }
}


void pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr)
{
    u32 inputStride = inputRows * inputCols * inputComponents;
    u32 outputStride = (inputRows / poolRows) * (inputCols / poolCols) * inputComponents;
    for (u32 i = 0; i < inputCount; i++)
    {
        s_pool2d(inputPtr + i*inputStride, inputRows, inputCols, inputComponents,
                                           poolRows, poolCols,
                 outputPtr + i*outputStride);
    }
}


void un_pool2d_multi_input(
        u32 inputCount,
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
        const fml* srcPtr,
              fml* destPtr)
{
    // TODO
}


}  // namespace cpu_golden
}  // namespace pool2d
}  // namespace ml
