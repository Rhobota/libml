#include <ml/pool2d/cpu_golden.h>

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


#define RUN_POOL2d_TEST \
    fml* output = new fml[outputSize+100]; \
    for (u32 i = 0; i < outputSize+100; i++) \
        output[i] = FML(12345.0); \
    ml::pool2d::cpu_golden::pool2d_multi_input(inputCount, input, inputRows, inputCols, inputComponents, poolRows, poolCols, output); \
    for (u32 i = 0; i < outputSize; i++) \
    { \
        if (output[i] != expectedOutput[i]) \
        { \
            std::cerr << "FAILED AT OUTPUT " << i << ": expected " << expectedOutput[i] << ", but got " << output[i] << std::endl; \
            t.fail(); \
        } \
    } \
    for (u32 i = outputSize; i < outputSize+100; i++) \
    { \
        if (output[i] != FML(12345.0)) \
        { \
            std::cerr << "BUFFER OVERRUN AT INDEX " << i << std::endl; \
            t.fail(); \
        } \
    } \


void pool2d_test0(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), };
    u32 inputRows = 1;
    u32 inputCols = 1;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("pool2d() golden test 0", pool2d_test0);

    return 0;
}
