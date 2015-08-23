#include <ml/tWrappedGPULayer.h>

#include "cuda_stuff.ipp"


namespace ml
{


tWrappedGPULayer::tWrappedGPULayer(u32 numInputDims, u32 numOutputDims, iLayer* wrappedLayer)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_wrappedLayer(wrappedLayer),
      m_curCount(0),
      m_maxCount(0),
      m_gpu_input(NULL),
      m_output(NULL),
      m_gpu_outputErrorGradients(NULL),
      m_inputErrorGradients(NULL)
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

    s_cudaFree(m_gpu_input);
    s_cudaFree(m_gpu_outputErrorGradients);

    delete [] m_output;
    delete [] m_inputErrorGradients;
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
    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims");

    if (count == 0)
        throw eInvalidArgument("count must be positive.");

    if (count > m_maxCount)
    {
        s_cudaFree(m_gpu_input);
        s_cudaFree(m_gpu_outputErrorGradients);

        delete [] m_output;
        delete [] m_inputErrorGradients;

        m_gpu_input                = s_cudaMalloc(m_numInputDims * count);
        m_gpu_outputErrorGradients = s_cudaMalloc(m_numOutputDims * count);

        m_output              = new fml[m_numOutputDims * count];
        m_inputErrorGradients = new fml[m_numInputDims * count];

        m_maxCount = count;
    }
    m_curCount = count;

    s_cudaCopyHostToDevice(m_gpu_input, input, numInputDims*count);
    m_wrappedLayer->takeInput(m_gpu_input, numInputDims, count);

    u32 retNumOutputDims = 0, retCount = 0;
    const fml* gpu_output = m_wrappedLayer->getOutput(retNumOutputDims, retCount);
    if (retNumOutputDims != m_numOutputDims)
        throw eRuntimeError("Unexpected retNumOutputDims from wrapped layer.");
    if (retCount != count)
        throw eRuntimeError("Unexpected retCount from wrapped layer.");
    s_cudaCopyDeviceToHost(m_output, gpu_output, m_numOutputDims*count);
}

const fml* tWrappedGPULayer::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_output;
}

void tWrappedGPULayer::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    if (!outputErrorGradients)
        throw eInvalidArgument("outputErrorGradients may not be null!");

    if (numOutputDims != m_numOutputDims)
        throw eInvalidArgument("Unexpected numOutputDims");

    if (outputCount != m_curCount)
        throw eInvalidArgument("Unexpected outputCount");

    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims");

    if (inputCount != m_curCount)
        throw eInvalidArgument("Unexpected inputCount");

    if (m_curCount == 0 || !m_gpu_outputErrorGradients || !m_inputErrorGradients)
        throw eRuntimeError("What gives?");

    s_cudaCopyHostToDevice(m_gpu_outputErrorGradients, outputErrorGradients, numOutputDims*outputCount);
    m_wrappedLayer->takeOutputErrorGradients(
            m_gpu_outputErrorGradients, numOutputDims, outputCount,
            m_gpu_input, numInputDims, inputCount,
            calculateInputErrorGradients);

    u32 retNumInputDims = 0, retCount = 0;
    const fml* gpu_inputErrorGradients = m_wrappedLayer->getInputErrorGradients(retNumInputDims, retCount);
    if (retNumInputDims != m_numInputDims)
        throw eRuntimeError("Unexpected retNumInputDims from wrapped layer.");
    if (retCount != inputCount)
        throw eRuntimeError("Unexpected retCount from wrapped layer.");
    s_cudaCopyDeviceToHost(m_inputErrorGradients, gpu_inputErrorGradients, numInputDims*inputCount);
}

const fml* tWrappedGPULayer::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_inputErrorGradients;
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
