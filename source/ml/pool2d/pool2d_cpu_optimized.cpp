#include <ml/pool2d/cpu_optimized.h>

#define NDEBUG 1
#include "../Eigen.h"


namespace ml
{
namespace pool2d
{
namespace cpu_optimized
{


static
void s_pool2d(
        const fml* inputPtr,  u32 inputRows,  u32 inputCols,  u32 inputComponents,
                              u32 poolRows,  u32 poolCols,
              fml* outputPtr)
{
    assert(inputPtr);
    assert(inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(poolRows > 0 && poolCols > 0);
    assert(outputPtr);

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


static
void s_un_pool2d(
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

    MapRowMajor dest(destPtr, inputRows, inputCols*inputComponents);
    dest.setZero();

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
                MapRowMajorWithStrideConst::Index row = 0, col = 0;
                input.block(r*poolRows, c*poolCols, poolRows, poolCols).maxCoeff(&row, &col);
                fml value = *srcPtr++;
                MapRowMajorWithStride dest(destPtr+i,
                                           inputRows, inputCols,
                                           Stride(inputCols*inputComponents, inputComponents));
                dest(r*poolRows + row, c*poolCols + col) = value;
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

    #ifdef LIBML_HAS_OPENMP
    #pragma omp parallel for
    #endif
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
    u32 inputStride = inputRows * inputCols * inputComponents;
    u32 outputStride = (inputRows / poolRows) * (inputCols / poolCols) * inputComponents;

    #ifdef LIBML_HAS_OPENMP
    #pragma omp parallel for
    #endif
    for (u32 i = 0; i < inputCount; i++)
    {
        s_un_pool2d(inputPtr + i*inputStride, inputRows, inputCols, inputComponents,
                                              poolRows, poolCols,
                    srcPtr + i*outputStride,
                    destPtr + i*inputStride);
    }
}


}  // namespace cpu_optimized
}  // namespace pool2d
}  // namespace ml
