#include <ml2/tCnnLayerCPU.h>

#include "common.ipp"

#include "../ml/Eigen.h"


namespace ml2
{


tCnnLayerCPU::tCnnLayerCPU()
    : tAnnLayerCPU()
{
}


tCnnLayerCPU::tCnnLayerCPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                           u32 numInputRows, u32 numInputCols, u32 numInputComponents,
                           u32 numFeatureMapRows, u32 numFeatureMapCols, u32 numFeatureMaps,
                           algo::iLCG& lcg, fml randWeightMin, fml randWeightMax)
    : tAnnLayerCPU(type, rule, numInputDims, numNeurons, lcg,
                   randWeightMin, randWeightMax)
{
}


tCnnLayerCPU::~tCnnLayerCPU()
{
    // The super d'tor is called automatically.
}


void tCnnLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    // TODO
}


const fml* tCnnLayerCPU::getOutput(u32& numOutputDims, u32& count) const
{
    // TODO
}


void tCnnLayerCPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    // TODO
}


const fml* tCnnLayerCPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    // TODO
}


static
iLayer* s_newLayerFunc(iReadable* in)
{
    tCnnLayerCPU* layer = new tCnnLayerCPU();
    layer->unpack(in);
    return layer;
}


static u32 layerId = 78438;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tCnnLayerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


void tCnnLayerCPU::reset()
{
    // TODO
}


}   // namespace ml2
