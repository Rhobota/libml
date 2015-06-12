#include <ml/conv2d/cpu_optimized.h>


namespace ml
{
namespace conv2d
{
namespace cpu_optimized
{


static
void s_conv2d(
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
    // TODO: templatize -- including templatizing the block() calls
    // TODO: re-structure to remove as many ifs from the inner loops as possible

    assert(inputPtr && inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(kernelPtr && (kernelRows % 2) == 1 && (kernelCols % 2) == 1);
    assert(kernelStepY > 0 && kernelStepX > 0 && numKernels > 0);
    assert(kernelBiases);
    assert(outputPtr);

    u32 kernelRadiusY = kernelRows / 2;
    u32 kernelRadiusX = kernelCols / 2;

    MapRowMajorConst input(inputPtr, inputRows, inputCols * inputComponents);

    for (u32 r = 0; r < inputRows; r += kernelStepY)
    {
        u32 y, ky;
        if (r < kernelRadiusY)
            y = 0, ky = kernelRadiusY - r;
        else
            y = r - kernelRadiusY, ky = 0;

        u32 h = r + kernelRadiusY + 1;  // not really "+ 1", but "+ (kernelRows%2)", except we know that is always 1, so we're taking a shortcut here
        if (h > inputRows)
            h = inputRows;
        h -= y;

        for (u32 c = 0; c < inputCols; c += kernelStepX)
        {
            u32 x, kx;
            if (c < kernelRadiusX)
                x = 0, kx = kernelRadiusX - c;
            else
                x = c - kernelRadiusX, kx = 0;

            u32 w = c + kernelRadiusX + 1;  // not really "+ 1", but "+ (kernelCols%2)", except we know that is always 1, so we're taking a shortcut here
            if (w > inputCols)
                w = inputCols;
            w -= x;

            for (u32 i = 0; i < numKernels; i++)
            {
                MapRowMajorConst kernel(kernelPtr + i * kernelRows * kernelCols * inputComponents,
                                        kernelRows,
                                        kernelCols * inputComponents);
                fml val = input.block(y, x*inputComponents, h, w*inputComponents)
                               .cwiseProduct(kernel.block(ky, kx*inputComponents, h, w*inputComponents))
                               .sum();
                *outputPtr++ = (val + kernelBiases[i]) * scaleFactor;
            }
        }
    }
}


static
void s_conv2d_backprop(
              fml* di_ptr,    u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
        const fml* dA_ptr)
{
    // TODO: templatize -- including templatizing the block() calls
    // TODO: re-structure to remove as many ifs from the inner loops as possible

    assert(di_ptr && inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(kernelPtr && (kernelRows % 2) == 1 && (kernelCols % 2) == 1);
    assert(kernelStepY > 0 && kernelStepX > 0 && numKernels > 0);
    assert(kernelBiases);
    assert(dA_ptr);

    u32 kernelRadiusY = kernelRows / 2;
    u32 kernelRadiusX = kernelCols / 2;

    MapRowMajor di(di_ptr, inputRows, inputCols * inputComponents);
    di.setZero();

    for (u32 r = 0; r < inputRows; r += kernelStepY)
    {
        u32 y, ky;
        if (r < kernelRadiusY)
            y = 0, ky = kernelRadiusY - r;
        else
            y = r - kernelRadiusY, ky = 0;

        u32 h = r + kernelRadiusY + 1;  // not really "+ 1", but "+ (kernelRows%2)", except we know that is always 1, so we're taking a shortcut here
        if (h > inputRows)
            h = inputRows;
        h -= y;

        for (u32 c = 0; c < inputCols; c += kernelStepX)
        {
            u32 x, kx;
            if (c < kernelRadiusX)
                x = 0, kx = kernelRadiusX - c;
            else
                x = c - kernelRadiusX, kx = 0;

            u32 w = c + kernelRadiusX + 1;  // not really "+ 1", but "+ (kernelCols%2)", except we know that is always 1, so we're taking a shortcut here
            if (w > inputCols)
                w = inputCols;
            w -= x;

            for (u32 i = 0; i < numKernels; i++)
            {
                MapRowMajorConst kernel(kernelPtr + i * kernelRows * kernelCols * inputComponents,
                                        kernelRows,
                                        kernelCols * inputComponents);
                fml dA = *dA_ptr++;
                dA *= scaleFactor;
                di.block(y, x*inputComponents, h, w*inputComponents) += kernel.block(ky, kx*inputComponents, h, w*inputComponents) * dA;
            }
        }
    }
}


static
void s_conv2d_accumError(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr, u32 inputRows,   u32 inputCols,  u32 inputComponents,
              fml* dk_ptr,   u32 kernelRows,  u32 kernelCols,
                             u32 kernelStepY, u32 kernelStepX,
                             u32 numKernels, u32 kernelIndex,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    // TODO: re-structure to remove as many ifs from the inner loops as possible

    assert(inputPtr && inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(dk_ptr && (kernelRows % 2) == 1 && (kernelCols % 2) == 1);
    assert(kernelStepY > 0 && kernelStepX > 0);
    assert(numKernels > 0 && kernelIndex < numKernels);
    assert(db_ptr);
    assert(dA_ptr);

    u32 kernelRadiusY = kernelRows / 2;
    u32 kernelRadiusX = kernelCols / 2;

    u32 outputRows = (inputRows - 1) / kernelStepY + 1;
    u32 outputCols = (inputCols - 1) / kernelStepX + 1;

    fml* sum = new fml[inputComponents];

    *db_ptr = FML(0.0);

    {
        MapRowMajor dk_map(dk_ptr, kernelRows, kernelCols*inputComponents);
        dk_map.setZero();
    }

    for (u32 inputIndex = 0; inputIndex < inputCount; inputIndex++)
    {
        const fml* input_start = inputPtr + inputIndex*inputStride;
        const fml* dA_start = dA_ptr + inputIndex*outputStride + kernelIndex;

        {
            MapRowMajorWithStrideConst dA_map(dA_start, outputRows, outputCols,
                                              Stride(outputCols*numKernels, numKernels));
            *db_ptr += dA_map.sum() * scaleFactor;
        }

        for (u32 kernelRowIndex = 0; kernelRowIndex < kernelRows; kernelRowIndex++)
        {
            fml* dk_row = dk_ptr + kernelRowIndex*kernelCols*inputComponents;

            for (u32 kernelColIndex = 0; kernelColIndex < kernelCols; kernelColIndex++)
            {
                fml* dk_here = dk_row + kernelColIndex*inputComponents;

                for (u32 inputComponentIndex = 0; inputComponentIndex < inputComponents; inputComponentIndex++)
                    sum[inputComponentIndex] = FML(0.0);

                for (u32 inputRowIndex = 0, dA_rowIndex = 0; inputRowIndex < inputRows; inputRowIndex += kernelStepY, dA_rowIndex++)
                {
                    u32 rowOfInterest;
                    if (kernelRowIndex < kernelRadiusY)
                    {
                        if (inputRowIndex < (kernelRadiusY - kernelRowIndex))
                            continue;
                        rowOfInterest = inputRowIndex - (kernelRadiusY - kernelRowIndex);
                    }
                    else
                    {
                        rowOfInterest = inputRowIndex + (kernelRowIndex - kernelRadiusY);
                        if (rowOfInterest >= inputRows)
                            continue;
                    }

                    const fml* input_row = input_start + rowOfInterest*inputCols*inputComponents;
                    const fml* dA_row = dA_start + dA_rowIndex*outputCols*numKernels;

                    for (u32 inputColIndex = 0, dA_colIndex = 0; inputColIndex < inputCols; inputColIndex += kernelStepX, dA_colIndex++)
                    {
                        u32 colOfInterest;
                        if (kernelColIndex < kernelRadiusX)
                        {
                            if (inputColIndex < (kernelRadiusX - kernelColIndex))
                                continue;
                            colOfInterest = inputColIndex - (kernelRadiusX - kernelColIndex);
                        }
                        else
                        {
                            colOfInterest = inputColIndex + (kernelColIndex - kernelRadiusX);
                            if (colOfInterest >= inputCols)
                                continue;
                        }

                        const fml* input_here = input_row + colOfInterest*inputComponents;
                        fml dA_here = dA_row[dA_colIndex*numKernels];

                        for (u32 inputComponentIndex = 0; inputComponentIndex < inputComponents; inputComponentIndex++)
                        {
                            sum[inputComponentIndex] += input_here[inputComponentIndex] * dA_here;
                        }
                    }
                }

                for (u32 inputComponentIndex = 0; inputComponentIndex < inputComponents; inputComponentIndex++)
                    dk_here[inputComponentIndex] += sum[inputComponentIndex] * scaleFactor;
            }
        }
    }

    delete [] sum;
}


void conv2d_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
    #ifdef LIBML_HAS_OPENMP
    #pragma omp parallel for
    #endif
    for (u32 i = 0; i < inputCount; i++)
    {
        s_conv2d(inputPtr + i*inputStride, inputRows, inputCols, inputComponents,
                 kernelPtr, kernelRows, kernelCols,
                            kernelStepY, kernelStepX,
                            numKernels,
                 kernelBiases, scaleFactor,
                 outputPtr + i*outputStride);
    }
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
    #ifdef LIBML_HAS_OPENMP
    #pragma omp parallel for
    #endif
    for (u32 i = 0; i < inputCount; i++)
    {
        s_conv2d_backprop(di_ptr + i*inputStride, inputRows, inputCols, inputComponents,
                          kernelPtr, kernelRows, kernelCols,
                                     kernelStepY, kernelStepX,
                                     numKernels,
                          kernelBiases, scaleFactor,
                          dA_ptr + i*outputStride);
    }
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
    u32 kernelDims = kernelRows*kernelCols*inputComponents;

    #ifdef LIBML_HAS_OPENMP
    #pragma omp parallel for
    #endif
    for (u32 i = 0; i < numKernels; i++)
    {
        s_conv2d_accumError(
                inputCount, inputStride, outputStride,
                inputPtr, inputRows, inputCols, inputComponents,
                dk_ptr + i*kernelDims, kernelRows, kernelCols,
                                       kernelStepY, kernelStepX,
                                       numKernels, i,
                db_ptr + i, scaleFactor,
                dA_ptr);
    }
}


}  // namespace cpu_optimized
}  // namespace conv2d
}  // namespace ml
