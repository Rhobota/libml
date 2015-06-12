/*
 * This file has the gold standard implementation of these three functions.
 * Any new implementations should be tested against these.
 */


namespace ml
{
namespace    // <-- un-named namespaces act like everything inside is statically scoped
{


void s_conv2d(
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
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


void s_conv2d_backprop(
              fml* di_ptr,    u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
        const fml* dA_ptr)
{
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


void s_conv2d_accumError(
        const fml* inputPtr, u32 inputRows,   u32 inputCols,   u32 inputComponents,
              fml* dk_ptr,   u32 kernelRows,  u32 kernelCols,
                             u32 kernelStepY, u32 kernelStepX,
                             u32 numKernels,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    assert(inputPtr && inputRows > 0 && inputCols > 0 && inputComponents > 0);
    assert(dk_ptr && (kernelRows % 2) == 1 && (kernelCols % 2) == 1);
    assert(kernelStepY > 0 && kernelStepX > 0 && numKernels > 0);
    assert(db_ptr);
    assert(dA_ptr);

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
                MapRowMajor dk(dk_ptr + i * kernelRows * kernelCols * inputComponents,
                               kernelRows,
                               kernelCols * inputComponents);
                fml dA = *dA_ptr++;
                dA *= scaleFactor;
                dk.block(ky, kx*inputComponents, h, w*inputComponents) += input.block(y, x*inputComponents, h, w*inputComponents) * dA;
                db_ptr[i] += dA;
            }
        }
    }
}


void s_conv2d_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr,  u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
              fml* outputPtr)
{
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


void s_conv2d_backprop_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
              fml* di_ptr,    u32 inputRows,   u32 inputCols,   u32 inputComponents,
        const fml* kernelPtr, u32 kernelRows,  u32 kernelCols,
                              u32 kernelStepY, u32 kernelStepX,
                              u32 numKernels,
        const fml* kernelBiases, fml scaleFactor,
        const fml* dA_ptr)
{
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


void s_conv2d_accumError_multi_input(
        u32 inputCount,  u32 inputStride,  u32 outputStride,
        const fml* inputPtr, u32 inputRows,   u32 inputCols,   u32 inputComponents,
              fml* dk_ptr,   u32 kernelRows,  u32 kernelCols,
                             u32 kernelStepY, u32 kernelStepX,
                             u32 numKernels,
              fml* db_ptr, fml scaleFactor,
        const fml* dA_ptr)
{
    u32 kernelDims = kernelRows*kernelCols*inputComponents*numKernels;

    Map dk(dk_ptr, kernelDims, 1);
    Map db(db_ptr, numKernels, 1);

    dk.setZero();
    db.setZero();

    for (u32 i = 0; i < inputCount; i++)
    {
        s_conv2d_accumError(inputPtr + i*inputStride, inputRows, inputCols, inputComponents,
                            dk_ptr, kernelRows, kernelCols,
                                    kernelStepY, kernelStepX,
                                    numKernels,
                            db_ptr, scaleFactor,
                            dA_ptr + i*outputStride);
    }
}


}  // end of anonymous namespace
}  // end of namespace ml
