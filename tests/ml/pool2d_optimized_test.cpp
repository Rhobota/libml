#include <ml/pool2d/cpu_golden.h>
#include <ml/pool2d/cpu_optimized.h>

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


#define dout if (false) std::cout


static const int kTestIterations = 3000;


void pool2dTest(
        const tTest& t,

        u32 inputRows,
        u32 inputCols,
        u32 inputComponents,
        u32 numInputs,

        u32 poolRows,
        u32 poolCols)
{
    fml* input = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        input[i] = rand() % 100;

    u32 outputRows = inputRows / poolRows;
    u32 outputCols = inputCols / poolCols;

    fml* output1 = new fml[outputRows*outputCols*inputComponents*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*inputComponents*numInputs; i++)
        output1[i] = FML(1000.0);

    fml* output2 = new fml[outputRows*outputCols*inputComponents*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*inputComponents*numInputs; i++)
        output2[i] = FML(700000.0);

    ml::pool2d::cpu_optimized::pool2d_multi_input(
            numInputs,
            input, inputRows, inputCols, inputComponents,
                     poolRows, poolCols,
            output1);

    dout << "output1:" << std::endl;
    for (u32 r = 0; r < outputRows*numInputs; r++)
    {
        for (u32 c = 0; c < outputCols*inputComponents; c++)
        {
            dout << ' ' << output1[r*outputCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    ml::pool2d::cpu_golden::pool2d_multi_input(
            numInputs,
            input, inputRows, inputCols, inputComponents,
                     poolRows, poolCols,
            output2);

    dout << "output2:" << std::endl;
    for (u32 r = 0; r < outputRows*numInputs; r++)
    {
        for (u32 c = 0; c < outputCols*inputComponents; c++)
        {
            dout << ' ' << output2[r*outputCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    for (u32 r = 0; r < outputRows*numInputs; r++)
    {
        for (u32 c = 0; c < outputCols*inputComponents; c++)
        {
            fml a = output1[r*outputCols*inputComponents + c];
            fml b = output2[r*outputCols*inputComponents + c];
            if (fabs(a - b) > 0.00000001)
            {
                std::cerr << "NOT EQUAL! " << a << " != " << b << std::endl;
                t.fail();
            }
        }
    }

    delete [] output2;
    delete [] output1;

    delete [] input;
}


void pool2dTest(const tTest& t)
{
    for (int i = 0; i < kTestIterations; i++)
    {
        u32 inputRows = (rand() % 100) + 1;
        u32 inputCols = (rand() % 100) + 1;
        u32 inputComponents = (rand() % 15) + 1;
        u32 numInputs = (rand() % 10) + 1;

        u32 poolRows = 2*(rand() % 4) + 1;
        u32 poolCols = 2*(rand() % 4) + 1;

        dout << "inputRows = " << inputRows << std::endl;
        dout << "inputCols = " << inputCols << std::endl;
        dout << "inputComponents = " << inputComponents << std::endl;
        dout << "numInputs = " << numInputs << std::endl;

        dout << "poolRows = " << poolRows << std::endl;
        dout << "poolCols = " << poolCols << std::endl;

        pool2dTest(
                t,

                inputRows,
                inputCols,
                inputComponents,
                numInputs,

                poolRows,
                poolCols);
    }
}


void un_pool2dTest(
        const tTest& t,

        u32 inputRows,
        u32 inputCols,
        u32 inputComponents,
        u32 numInputs,

        u32 poolRows,
        u32 poolCols)
{
    fml* input = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        input[i] = rand() % 100;

    fml* dest1 = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        dest1[i] = FML(1000.0);

    fml* dest2 = new fml[inputRows*inputCols*inputComponents*numInputs];
    for (u32 i = 0; i < inputRows*inputCols*inputComponents*numInputs; i++)
        dest2[i] = FML(700000.0);

    u32 outputRows = inputRows / poolRows;
    u32 outputCols = inputCols / poolCols;

    fml* src = new fml[outputRows*outputCols*inputComponents*numInputs];
    for (u32 i = 0; i < outputRows*outputCols*inputComponents*numInputs; i++)
        src[i] = rand() % 100;

    ml::pool2d::cpu_optimized::un_pool2d_multi_input(
            numInputs,
            input, inputRows, inputCols, inputComponents,
                     poolRows, poolCols,
            src,
            dest1);

    dout << "dest1:" << std::endl;
    for (u32 r = 0; r < inputRows*numInputs; r++)
    {
        for (u32 c = 0; c < inputCols*inputComponents; c++)
        {
            dout << ' ' << dest1[r*inputCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    ml::pool2d::cpu_golden::un_pool2d_multi_input(
            numInputs,
            input, inputRows, inputCols, inputComponents,
                     poolRows, poolCols,
            src,
            dest2);

    dout << "dest2:" << std::endl;
    for (u32 r = 0; r < inputRows*numInputs; r++)
    {
        for (u32 c = 0; c < inputCols*inputComponents; c++)
        {
            dout << ' ' << dest2[r*inputCols*inputComponents + c];
        }
        dout << std::endl;
    }
    dout << std::endl;

    for (u32 r = 0; r < inputRows*numInputs; r++)
    {
        for (u32 c = 0; c < inputCols*inputComponents; c++)
        {
            fml a = dest1[r*inputCols*inputComponents + c];
            fml b = dest2[r*inputCols*inputComponents + c];
            if (fabs(a - b) > 0.00000001)
            {
                std::cerr << "NOT EQUAL! " << r << "x" << c << "    " << a << " != " << b << std::endl;
                t.fail();
            }
        }
    }

    delete [] src;

    delete [] dest2;
    delete [] dest1;

    delete [] input;
}


void un_pool2dTest(const tTest& t)
{
    for (int i = 0; i < kTestIterations; i++)
    {
        u32 inputRows = (rand() % 100) + 1;
        u32 inputCols = (rand() % 100) + 1;
        u32 inputComponents = (rand() % 15) + 1;
        u32 numInputs = (rand() % 10) + 1;

        u32 poolRows = 2*(rand() % 4) + 1;
        u32 poolCols = 2*(rand() % 4) + 1;

        dout << "inputRows = " << inputRows << std::endl;
        dout << "inputCols = " << inputCols << std::endl;
        dout << "inputComponents = " << inputComponents << std::endl;
        dout << "numInputs = " << numInputs << std::endl;

        dout << "poolRows = " << poolRows << std::endl;
        dout << "poolCols = " << poolCols << std::endl;

        un_pool2dTest(
                t,

                inputRows,
                inputCols,
                inputComponents,
                numInputs,

                poolRows,
                poolCols);
    }
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("pool2d() optimized test", pool2dTest);
    tTest("un_pool2d() optimized test", un_pool2dTest);

    return 0;
}
