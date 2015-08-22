#include <ml/tSplittingLayerCPU.h>
#include <ml/tScalingLayerCPU.h>
#include <ml/tReductionLayerCPU.h>

#include <rho/tTest.h>
#include <rho/tCrashReporter.h>

using namespace rho;
using ml::fml;


#define dout if (false) std::cout


static const int kTestIterations = 100;
static const int kTestInnerIterations = 100;


void test1(const tTest& t)
{
    u32 numInputDims = (rand() % 20) + 1;
    u32 numOutputDims = numInputDims;
    ml::tSplittingLayerCPU layer(numInputDims, numOutputDims);

    std::vector< std::pair<u32, fml> > layerInfo;

    u32 sum = 0;
    while (sum < numInputDims)
    {
        u32 innerInputDims = (rand() % (numInputDims - sum)) + 1;
        u32 innerOutputDims = innerInputDims;
        fml scaleFactor = (rand() % 50);
        ml::iLayer* innerLayer = new ml::tScalingLayerCPU(innerInputDims, innerOutputDims, scaleFactor);
        layer.addLayer(innerLayer, innerInputDims, innerOutputDims);
        sum += innerInputDims;
        layerInfo.push_back(std::make_pair(innerInputDims, scaleFactor));
    }
    t.assert(sum == numInputDims);
    t.assert(sum == numOutputDims);

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
            {
                u32 index = 0;
                for (size_t i = 0; i < layerInfo.size(); i++)
                {
                    for (u32 o = 0; o < layerInfo[i].first; o++)
                    {
                        correctOutput[c*numOutputDims + index] = input[c*numOutputDims + index] * layerInfo[i].second;
                        index++;
                    }
                }
                t.assert(index == numOutputDims);
            }

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
            {
                u32 index = 0;
                for (size_t i = 0; i < layerInfo.size(); i++)
                {
                    for (u32 o = 0; o < layerInfo[i].first; o++)
                    {
                        correctInError[c*numInputDims + index] = outError[c*numInputDims + index] * layerInfo[i].second;
                        index++;
                    }
                }
                t.assert(index == numInputDims);
            }

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


void test2(const tTest& t)
{
    u32 numInputDims = (rand() % 20) + 1;

    std::vector<u32> layerInfo;

    u32 sum = 0;
    while (sum < numInputDims)
    {
        u32 innerInputDims = (rand() % (numInputDims - sum)) + 1;
        sum += innerInputDims;
        layerInfo.push_back(innerInputDims);
    }
    t.assert(sum == numInputDims);

    u32 numOutputDims = layerInfo.size();
    ml::tSplittingLayerCPU layer(numInputDims, numOutputDims);

    for (size_t i = 0; i < layerInfo.size(); i++)
    {
        u32 innerInputDims = layerInfo[i];
        u32 innerOutputDims = 1;
        ml::iLayer* innerLayer = new ml::tReductionLayerCPU(innerInputDims, innerOutputDims);
        layer.addLayer(innerLayer, innerInputDims, innerOutputDims);
    }

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
            {
                u32 index = 0;
                for (size_t i = 0; i < layerInfo.size(); i++)
                {
                    fml sum = FML(0.0);
                    for (u32 o = 0; o < layerInfo[i]; o++)
                    {
                        sum += input[c*numInputDims + index];
                        index++;
                    }
                    correctOutput[c*numOutputDims + i] = sum;
                }
                t.assert(index == numInputDims);
                t.assert(layerInfo.size() == numOutputDims);
            }

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
            {
                u32 index = 0;
                for (size_t i = 0; i < layerInfo.size(); i++)
                {
                    for (u32 o = 0; o < layerInfo[i]; o++)
                    {
                        correctInError[c*numInputDims + index] = outError[c*numOutputDims + i];
                        index++;
                    }
                }
                t.assert(index == numInputDims);
                t.assert(layerInfo.size() == numOutputDims);
            }

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

    tTest("splitting layer test (1)", test1, kTestIterations);
    tTest("splitting layer test (2)", test2, kTestIterations);

    return 0;
}
