#include <ml/tScalingLayerCPU.h>

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
    fml scaleFactor = (rand() % 100) + 1;
    ml::tScalingLayerCPU layer(numInputDims, numOutputDims, scaleFactor);
    for (int iter = 0; iter < kTestInnerIterations; iter++)
    {
        // Run some test values through the scaling layer.
    }
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("scaling layer test", test, kTestIterations);

    return 0;
}
