#include <ml/tWrappedGPULayer.h>

#include "cuda_stuff.ipp"


namespace ml
{


tWrappedGPULayer::tWrappedGPULayer(u32 numInputDims, u32 numOutputDims, iLayer* wrappedLayer)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_wrappedLayer(wrappedLayer),
      m_curCount(0),
      m_maxCount(0)
{
    if (numInputDims == 0)
        throw eInvalidArgument("numInputDims must be positive!");
    if (numOutputDims == 0)
        throw eInvalidArgument("numOutputDims must be positive!");
    if (!wrappedLayer)
        throw eInvalidArgument("wrappedLayer must not be NULL.");
}

tWrappedGPULayer::~tWrappedGPULayer()
{
    delete m_wrappedLayer;
    m_wrappedLayer = NULL;
}

fml tWrappedGPULayer::calculateError(const tIO& output, const tIO& target)
{
    return m_wrappedLayer->calculateError(output, target);
}

fml tWrappedGPULayer::calculateError(const std::vector<tIO>& outputs,
                                      const std::vector<tIO>& targets)
{
    return m_wrappedLayer->calculateError(outputs, targets);
}

void tWrappedGPULayer::reset()
{
    m_wrappedLayer->reset();
}

void tWrappedGPULayer::printLayerInfo(std::ostream& out) const
{
    m_wrappedLayer->printLayerInfo(out);
}

std::string tWrappedGPULayer::layerInfoString() const
{
    return m_wrappedLayer->layerInfoString();
}

void tWrappedGPULayer::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    // TODO
}

const fml* tWrappedGPULayer::getOutput(u32& numOutputDims, u32& count) const
{
    // TODO
    return NULL;
}

void tWrappedGPULayer::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    // TODO
}

const fml* tWrappedGPULayer::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    // TODO
    return NULL;
}

u32 tWrappedGPULayer::headerId() const
{
    throw eLogicError("Don't call headerId() on tWrappedGPULayer.");
}

void tWrappedGPULayer::pack(iWritable* out) const
{
    throw eLogicError("Don't call pack() on tWrappedGPULayer.");
}

void tWrappedGPULayer::unpack(iReadable* in)
{
    throw eLogicError("Don't call unpack() on tWrappedGPULayer.");
}


}   // namespace ml
