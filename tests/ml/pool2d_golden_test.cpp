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

    u32 poolRows = 1;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


void pool2d_test1(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5) };
    u32 inputRows = 1;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 1;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(2.5) };
    u32 outputSize = 1;

    RUN_POOL2d_TEST
}


void pool2d_test2(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5) };
    u32 inputRows = 2;
    u32 inputCols = 1;
    u32 inputComponents = 1;

    u32 poolRows = 1;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


void pool2d_test3(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5), FML(0.5), FML(-9.5) };
    u32 inputRows = 2;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 1;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(2.5), FML(0.5) };
    u32 outputSize = 2;

    RUN_POOL2d_TEST
}


void pool2d_test4(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), };
    u32 inputRows = 1;
    u32 inputCols = 1;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 1;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


void pool2d_test5(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5) };
    u32 inputRows = 1;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 1;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


void pool2d_test6(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5) };
    u32 inputRows = 2;
    u32 inputCols = 1;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 1;

    fml expectedOutput[] = { FML(2.5) };
    u32 outputSize = 1;

    RUN_POOL2d_TEST
}


void pool2d_test7(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5), FML(0.5), FML(9.5) };
    u32 inputRows = 2;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 1;

    fml expectedOutput[] = { FML(1.5), FML(9.5) };
    u32 outputSize = 2;

    RUN_POOL2d_TEST
}


void pool2d_test8(const tTest& t)
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


void pool2d_test9(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5) };
    u32 inputRows = 1;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


void pool2d_test10(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5) };
    u32 inputRows = 2;
    u32 inputCols = 1;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(0.0) };
    u32 outputSize = 0;

    RUN_POOL2d_TEST
}


void pool2d_test11(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5), FML(0.5), FML(-9.5) };
    u32 inputRows = 2;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(2.5) };
    u32 outputSize = 1;

    RUN_POOL2d_TEST
}


void pool2d_test12(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5), FML(0.5), FML(-9.5), FML(100.0), FML(100.0) };
    u32 inputRows = 3;
    u32 inputCols = 2;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(2.5) };
    u32 outputSize = 1;

    RUN_POOL2d_TEST
}


void pool2d_test13(const tTest& t)
{
    u32 inputCount = 1;
    fml input[] = { FML(1.5), FML(2.5), FML(100.0), FML(-9.5), FML(11.5), FML(100.0) };
    u32 inputRows = 2;
    u32 inputCols = 3;
    u32 inputComponents = 1;

    u32 poolRows = 2;
    u32 poolCols = 2;

    fml expectedOutput[] = { FML(11.5) };
    u32 outputSize = 1;

    RUN_POOL2d_TEST
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("pool2d() golden test 0", pool2d_test0);
    tTest("pool2d() golden test 1", pool2d_test1);
    tTest("pool2d() golden test 2", pool2d_test2);
    tTest("pool2d() golden test 3", pool2d_test3);
    tTest("pool2d() golden test 4", pool2d_test4);
    tTest("pool2d() golden test 5", pool2d_test5);
    tTest("pool2d() golden test 6", pool2d_test6);
    tTest("pool2d() golden test 7", pool2d_test7);
    tTest("pool2d() golden test 8", pool2d_test8);
    tTest("pool2d() golden test 9", pool2d_test9);
    tTest("pool2d() golden test 10", pool2d_test10);
    tTest("pool2d() golden test 11", pool2d_test11);
    tTest("pool2d() golden test 12", pool2d_test12);
    tTest("pool2d() golden test 13", pool2d_test13);
//     tTest("pool2d() golden test 14", pool2d_test14);
//     tTest("pool2d() golden test 15", pool2d_test15);
//     tTest("pool2d() golden test 16", pool2d_test16);
//     tTest("pool2d() golden test 17", pool2d_test17);
//     tTest("pool2d() golden test 18", pool2d_test18);
//     tTest("pool2d() golden test 19", pool2d_test19);
//     tTest("pool2d() golden test 20", pool2d_test20);
//     tTest("pool2d() golden test 21", pool2d_test21);
//     tTest("pool2d() golden test 22", pool2d_test22);
//     tTest("pool2d() golden test 23", pool2d_test23);
//     tTest("pool2d() golden test 24", pool2d_test24);
//     tTest("pool2d() golden test 25", pool2d_test25);

    return 0;
}
