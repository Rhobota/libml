#include <ml/common.h>
#include "../../source/ml/Eigen.h"
#include "../../source/ml/conv2d/conv2d_cpu_golden.ipp"

namespace optimized
{
    using namespace rho;
    using namespace ml;
    #include "../../source/ml/conv2d/conv2d_cpu_optimized.ipp"
}

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


#define dout if (false) std::cout


void doit(
        const tTest& t,

        u32 inputRows,
        u32 inputCols,
        u32 inputComponents,
        u32 numInputs,

        u32 kernelRows,
        u32 kernelCols,
        u32 numKernels,

        u32 kernelStepY,
        u32 kernelStepX)
{
    fml* input = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        input[i] = rand() % 100;

    u32 outputRows = (inputRows - 1) / kernelStepY + 1;
    u32 outputCols = (inputCols - 1) / kernelStepX + 1;

    fml* da = new fml[outputRows*outputCols*numKernels*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*numKernels*numInputs; i++)
        da[i] = rand() % 100;

    fml* dk1 = new fml[kernelRows*kernelCols*inputComponents*numKernels];
    fml* dk2 = new fml[kernelRows*kernelCols*inputComponents*numKernels];
    fml* db1 = new fml[numKernels];
    fml* db2 = new fml[numKernels];

    for (u32 i = 0; i < kernelRows*kernelCols*inputComponents*numKernels; i++)
        dk1[i] = FML(10000.0);
    for (u32 i = 0; i < numKernels; i++)
        db1[i] = FML(200000.0);
    for (u32 i = 0; i < kernelRows*kernelCols*inputComponents*numKernels; i++)
        dk2[i] = FML(30000.0);
    for (u32 i = 0; i < numKernels; i++)
        db2[i] = FML(400000.0);

    optimized::ml::s_conv2d_accumError_multi_input(
            numInputs, inputRows*inputCols*inputComponents, outputRows*outputCols*numKernels,
            input, inputRows, inputCols, inputComponents,
            dk1, kernelRows, kernelCols,
                 kernelStepY, kernelStepX,
                 numKernels,
            db1, FML(0.5),
            da);

    dout << "dk1:" << std::endl;
    for (u32 r = 0; r < kernelRows*numKernels; r++)
    {
        for (u32 c = 0; c < kernelCols*inputComponents; c++)
        {
            dout << ' ' << dk1[r*kernelCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << "db1:";
    for (u32 i = 0; i < numKernels; i++)
        dout << " " << db1[i];
    dout << std::endl << std::endl;

    ml::s_conv2d_accumError_multi_input(
            numInputs, inputRows*inputCols*inputComponents, outputRows*outputCols*numKernels,
            input, inputRows, inputCols, inputComponents,
            dk2, kernelRows, kernelCols,
                 kernelStepY, kernelStepX,
                 numKernels,
            db2, FML(0.5),
            da);

    dout << "dk2:" << std::endl;
    for (u32 r = 0; r < kernelRows*numKernels; r++)
    {
        for (u32 c = 0; c < kernelCols*inputComponents; c++)
        {
            dout << ' ' << dk2[r*kernelCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << "db2:";
    for (u32 i = 0; i < numKernels; i++)
        dout << " " << db2[i];
    dout << std::endl << std::endl;

    for (u32 r = 0; r < kernelRows*numKernels; r++)
    {
        for (u32 c = 0; c < kernelCols*inputComponents; c++)
        {
            fml a = dk1[r*kernelCols*inputComponents + c];
            fml b = dk2[r*kernelCols*inputComponents + c];
            if (fabs(a - b) > 0.00000001)
            {
                std::cerr << "NOT EQUAL! " << a << " != " << b << std::endl;
                t.fail();
            }
        }
    }
    for (u32 i = 0; i < numKernels; i++)
    {
        fml a = db1[i];
        fml b = db2[i];
        if (fabs(a - b) > 0.00000001)
        {
            std::cerr << "NOT EQUAL! " << a << " != " << b << std::endl;
            t.fail();
        }
    }

    delete [] db2;
    delete [] db1;
    delete [] dk2;
    delete [] dk1;

    delete [] da;
    delete [] input;
}


void test(const tTest& t)
{
    for (int i = 0; i < 1000; i++)
    {
        u32 inputRows = (rand() % 15) + 1;
        u32 inputCols = (rand() % 15) + 1;
        u32 inputComponents = (rand() % 3) + 1;
        u32 numInputs = (rand() % 5) + 1;

        u32 kernelRows = 2*(rand() % 4) + 1;
        u32 kernelCols = 2*(rand() % 4) + 1;
        u32 numKernels = (rand() % 15) + 1;

        u32 kernelStepY = (rand() % 4) + 1;
        u32 kernelStepX = (rand() % 4) + 1;

        dout << "inputRows = " << inputRows << std::endl;
        dout << "inputCols = " << inputCols << std::endl;
        dout << "inputComponents = " << inputComponents << std::endl;
        dout << "numInputs = " << numInputs << std::endl;

        dout << "kernelRows = " << kernelRows << std::endl;
        dout << "kernelCols = " << kernelCols << std::endl;
        dout << "numKernels = " << numKernels << std::endl;

        dout << "kernelStepY = " << kernelStepY << std::endl;
        dout << "kernelStepX = " << kernelStepX << std::endl;

        doit(
                t,

                inputRows,
                inputCols,
                inputComponents,
                numInputs,

                kernelRows,
                kernelCols,
                numKernels,

                kernelStepY,
                kernelStepX);
    }
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("convolve 2d optimized test", test);

    return 0;
}
