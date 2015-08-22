#include <ml/tReductionLayerCPU.h>

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


#define dout if (false) std::cout


static const int kTestIterations = 100;
static const int kTestInnerIterations = 100;


void test(const tTest& t)
{
    u32 numInputDims = (rand() % 100) + 1;
    u32 numOutputDims = (rand() % 100) + 1;
    ml::tReductionLayerCPU layer(numInputDims, numOutputDims);
    for (int iter = 0; iter < kTestInnerIterations; iter++)
    {
        // Run some test values through the reduction layer.
    }
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("reduction layer test", test, kTestIterations);

    return 0;
}
