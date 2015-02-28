#include <ml2/tAnnLayer.h>


namespace ml2
{


tAnnLayer::tAnnLayer(nAnnLayerType type, nAnnLayerWeightUpdateRule rule)
{
}


void tAnnLayer::setAlpha(fml alpha)
{
}


void tAnnLayer::setViscosity(fml viscosity)
{
}


void tAnnLayer::takeInput(fml* input, u32 numInputDims, u32 count)
{
}


fml* tAnnLayer::getOutput(u32& numOutputDims, u32& count) const
{
    return NULL;
}


void tAnnLayer::takeOutputErrorGradients(
                  fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  fml* input, u32 numInputDims, u32 inputCount)
{
}


fml* tAnnLayer::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    return NULL;
}


void tAnnLayer::pack(iWritable* out) const
{
}


void tAnnLayer::unpack(iReadable* in)
{
}


}   // namespace ml2
