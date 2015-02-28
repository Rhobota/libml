#include <ml2/tAnnLayer.h>


namespace ml2
{


tAnnLayer::tAnnLayer(nAnnLayerType type, nAnnLayerWeightUpdateRule rule)
    : m_type(type),
      m_rule(rule),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_output(NULL),
      m_numOutputDims(0),
      m_outputCount(0),
      m_prev_da(NULL),
      m_numInputDims(0),
      m_inputCount(0)
{
}


void tAnnLayer::setAlpha(fml alpha)
{
    if (alpha <= FML(0.0))
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_alpha = alpha;
}


void tAnnLayer::setViscosity(fml viscosity)
{
    if (viscosity <= FML(0.0) || viscosity >= FML(1.0))
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_viscosity = viscosity;
}


void tAnnLayer::takeInput(fml* input, u32 numInputDims, u32 count)
{
    // TODO
}


fml* tAnnLayer::getOutput(u32& numOutputDims, u32& count) const
{
    // TODO
    return NULL;
}


void tAnnLayer::takeOutputErrorGradients(
                  fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  fml* input, u32 numInputDims, u32 inputCount)
{
    // TODO
}


fml* tAnnLayer::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    // TODO
    return NULL;
}


void tAnnLayer::pack(iWritable* out) const
{
    // TODO
}


void tAnnLayer::unpack(iReadable* in)
{
    // TODO
}


}   // namespace ml2
