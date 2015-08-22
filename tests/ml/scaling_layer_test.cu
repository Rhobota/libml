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
    u32 numInputDims = (rand() % 20) + 1;
    u32 numOutputDims = numInputDims;
    fml scaleFactor = rand() % 100;
    ml::tScalingLayerCPU layer(numInputDims, numOutputDims, scaleFactor);

    for (int iter = 0; iter < kTestInnerIterations; iter++)
    {
        u32 count = (rand() % 10) + 1;
        fml* input          = new fml[numInputDims * count];
        fml* correctOutput  = new fml[numOutputDims * count];
        fml* outError       = new fml[numOutputDims * count];
        fml* correctInError = new fml[numInputDims * count];

        // Forward pass.
        {
            for (u32 c = 0; c < count; c++)
                for (u32 i = 0; i < numInputDims; i++)
                    input[c*numInputDims + i] = (rand() % 100) * (c + 1);

            for (u32 c = 0; c < count; c++)
                for (u32 o = 0; o < numOutputDims; o++)
                    correctOutput[c*numOutputDims + o] = input[c*numOutputDims + o] * scaleFactor;

            layer.takeInput(input, numInputDims, count);

            u32 retNumOutputDims = 0, retCount = 0;
            const fml* output = layer.getOutput(retNumOutputDims, retCount);
            t.assert(retNumOutputDims == numOutputDims);
            t.assert(retCount == count);

            for (u32 c = 0; c < count; c++)
            {
                for (u32 o = 0; o < numOutputDims; o++)
                {
                    if (fabs(correctOutput[c*numOutputDims + o] - output[c*numOutputDims + o]) > 0.00001)
                    {
                        std::cerr << correctOutput[c*numOutputDims + o] << "  " << output[c*numOutputDims + o] << std::endl;
                        t.fail();
                    }
                }
            }
        }

        // Backward pass.
        {
            for (u32 c = 0; c < count; c++)
                for (u32 o = 0; o < numOutputDims; o++)
                    outError[c*numOutputDims + o] = (rand() % 100) * (c + 1);

            for (u32 c = 0; c < count; c++)
                for (u32 i = 0; i < numInputDims; i++)
                    correctInError[c*numInputDims + i] = outError[c*numInputDims + i] * scaleFactor;

            layer.takeOutputErrorGradients(outError, numOutputDims, count,
                                           input, numInputDims, count,
                                           true);

            u32 retNumInputDims = 0, retCount = 0;
            const fml* inError = layer.getInputErrorGradients(retNumInputDims, retCount);
            t.assert(retNumInputDims == numInputDims);
            t.assert(retCount == count);

            for (u32 c = 0; c < count; c++)
            {
                for (u32 i = 0; i < numInputDims; i++)
                {
                    if (fabs(correctInError[c*numInputDims + i] - inError[c*numInputDims + i]) > 0.00001)
                    {
                        std::cerr << correctInError[c*numInputDims + i] << "  " << inError[c*numInputDims + i] << std::endl;
                        t.fail();
                    }
                }
            }
        }

        delete [] correctInError;
        delete [] outError;
        delete [] correctOutput;
        delete [] input;
    }
}


int main()
{
    tCrashReporter::init();

    srand(time(0));

    tTest("scaling layer test", test, kTestIterations);

    return 0;
}