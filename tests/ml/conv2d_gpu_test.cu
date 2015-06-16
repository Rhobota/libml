#include <ml/common.h>
#include <ml/conv2d/cpu_golden.h>
#include <ml/conv2d/gpu.h>

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


#define dout if (false) std::cout


static const int kTestIterations = 3000;


void convolveTest(
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

    fml* output1 = new fml[outputRows*outputCols*numKernels*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*numKernels*numInputs; i++)
        output1[i] = FML(1000.0);

    fml* output2 = new fml[outputRows*outputCols*numKernels*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*numKernels*numInputs; i++)
        output2[i] = FML(700000.0);

    fml* kernels = new fml[kernelRows*kernelCols*inputComponents*numKernels];
    fml* kernelBiases = new fml[numKernels];

    for (u32 i = 0; i < kernelRows*kernelCols*inputComponents*numKernels; i++)
        kernels[i] = rand() % 100;
    for (u32 i = 0; i < numKernels; i++)
        kernelBiases[i] = rand() % 100;

    ml::conv2d::gpu::conv2d_multi_input_with_memcpy(
            numInputs, inputRows*inputCols*inputComponents, outputRows*outputCols*numKernels,
            input, inputRows, inputCols, inputComponents,
            kernels, kernelRows, kernelCols,
                     kernelStepY, kernelStepX,
                     numKernels,
            kernelBiases, FML(0.5),
            output1);

    dout << "output1:" << std::endl;
    for (u32 r = 0; r < outputRows*numInputs; r++)
    {
        for (u32 c = 0; c < outputCols*numKernels; c++)
        {
            dout << ' ' << output1[r*outputCols*numKernels + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    ml::conv2d::cpu_golden::conv2d_multi_input(
            numInputs, inputRows*inputCols*inputComponents, outputRows*outputCols*numKernels,
            input, inputRows, inputCols, inputComponents,
            kernels, kernelRows, kernelCols,
                     kernelStepY, kernelStepX,
                     numKernels,
            kernelBiases, FML(0.5),
            output2);

    dout << "output2:" << std::endl;
    for (u32 r = 0; r < outputRows*numInputs; r++)
    {
        for (u32 c = 0; c < outputCols*numKernels; c++)
        {
            dout << ' ' << output2[r*outputCols*numKernels + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    for (u32 r = 0; r < outputRows*numInputs; r++)
    {
        for (u32 c = 0; c < outputCols*numKernels; c++)
        {
            fml a = output1[r*outputCols*numKernels + c];
            fml b = output2[r*outputCols*numKernels + c];
            if (fabs(a - b) > 0.00000001)
            {
                std::cerr << "NOT EQUAL! " << a << " != " << b << std::endl;
                t.fail();
            }
        }
    }

    delete [] kernelBiases;
    delete [] kernels;

    delete [] output2;
    delete [] output1;

    delete [] input;
}


void convolveTest(const tTest& t)
{
    for (int i = 0; i < kTestIterations; i++)
    {
        u32 inputRows = (rand() % 15) + 1;
        u32 inputCols = (rand() % 15) + 1;
        u32 inputComponents = (rand() % 3) + 1;
        u32 numInputs = (rand() % 5) + 1;

        u32 kernelRows = 2*(rand() % 3) + 3;
        u32 kernelCols = 2*(rand() % 3) + 3;
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

        convolveTest(
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


void backpropTest(
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
    fml* di1 = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        di1[i] = FML(400000.0);

    fml* di2 = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        di2[i] = FML(7000000.0);

    u32 outputRows = (inputRows - 1) / kernelStepY + 1;
    u32 outputCols = (inputCols - 1) / kernelStepX + 1;

    fml* da = new fml[outputRows*outputCols*numKernels*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*numKernels*numInputs; i++)
        da[i] = rand() % 100;

    fml* kernels = new fml[kernelRows*kernelCols*inputComponents*numKernels];
    fml* kernelBiases = new fml[numKernels];

    for (u32 i = 0; i < kernelRows*kernelCols*inputComponents*numKernels; i++)
        kernels[i] = rand() % 100;
    for (u32 i = 0; i < numKernels; i++)
        kernelBiases[i] = rand() % 100;

    ml::conv2d::gpu::conv2d_backprop_multi_input_with_memcpy(
            numInputs, inputRows*inputCols*inputComponents, outputRows*outputCols*numKernels,
            di1, inputRows, inputCols, inputComponents,
            kernels, kernelRows, kernelCols,
                     kernelStepY, kernelStepX,
                     numKernels,
            kernelBiases, FML(0.5),
            da);

    dout << "di1:" << std::endl;
    for (u32 r = 0; r < inputRows*numInputs; r++)
    {
        for (u32 c = 0; c < inputCols*inputComponents; c++)
        {
            dout << ' ' << di1[r*inputCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    ml::conv2d::cpu_golden::conv2d_backprop_multi_input(
            numInputs, inputRows*inputCols*inputComponents, outputRows*outputCols*numKernels,
            di2, inputRows, inputCols, inputComponents,
            kernels, kernelRows, kernelCols,
                     kernelStepY, kernelStepX,
                     numKernels,
            kernelBiases, FML(0.5),
            da);

    dout << "di2:" << std::endl;
    for (u32 r = 0; r < inputRows*numInputs; r++)
    {
        for (u32 c = 0; c < inputCols*inputComponents; c++)
        {
            dout << ' ' << di2[r*inputCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    for (u32 r = 0; r < inputRows*numInputs; r++)
    {
        for (u32 c = 0; c < inputCols*inputComponents; c++)
        {
            fml a = di1[r*inputCols*inputComponents + c];
            fml b = di2[r*inputCols*inputComponents + c];
            if (fabs(a - b) > 0.00000001)
            {
                std::cerr << "NOT EQUAL! " << a << " != " << b << std::endl;
                t.fail();
            }
        }
    }

    delete [] kernelBiases;
    delete [] kernels;

    delete [] da;

    delete [] di2;
    delete [] di1;
}


void backpropTest(const tTest& t)
{
    for (int i = 0; i < kTestIterations; i++)
    {
        u32 inputRows = (rand() % 15) + 1;
        u32 inputCols = (rand() % 15) + 1;
        u32 inputComponents = (rand() % 3) + 1;
        u32 numInputs = (rand() % 5) + 1;

        u32 kernelRows = 2*(rand() % 3) + 3;
        u32 kernelCols = 2*(rand() % 3) + 3;
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

        backpropTest(
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


void accumErrorTest(
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

    ml::conv2d::gpu::conv2d_accumError_multi_input_with_memcpy(
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

    ml::conv2d::cpu_golden::conv2d_accumError_multi_input(
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


void accumErrorTest(const tTest& t)
{
    for (int i = 0; i < kTestIterations; i++)
    {
        u32 inputRows = (rand() % 15) + 1;
        u32 inputCols = (rand() % 15) + 1;
        u32 inputComponents = (rand() % 3) + 1;
        u32 numInputs = (rand() % 5) + 1;

        u32 kernelRows = 2*(rand() % 3) + 3;
        u32 kernelCols = 2*(rand() % 3) + 3;
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

        accumErrorTest(
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

    tTest("conv2d() gpu test", convolveTest);

    //tTest("conv2d_backprop() gpu test", backpropTest);

    //tTest("conv2d_accumError() gpu test", accumErrorTest);

    return 0;
}
