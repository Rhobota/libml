#include <ml/tDropoutLayerCPU.h>


namespace ml
{


tDropoutLayerCPU::tDropoutLayerCPU()
    : tDropoutLayerBase(),
      m_output(NULL),
      m_inputErrorGradients(NULL)
{
}

tDropoutLayerCPU::tDropoutLayerCPU(u32 numInputDims, u32 numOutputDims, fml p)
    : tDropoutLayerBase(numInputDims, numOutputDims, p),
      m_output(NULL),
      m_inputErrorGradients(NULL)
{
}

tDropoutLayerCPU::~tDropoutLayerCPU()
{
    delete [] m_output;                 m_output = NULL;
    delete [] m_inputErrorGradients;    m_inputErrorGradients = NULL;
}

void tDropoutLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims");

    if (count == 0)
        throw eInvalidArgument("count must be positive.");

    if (count > m_maxCount)
    {
        delete [] m_output;                 m_output = NULL;
        delete [] m_inputErrorGradients;    m_inputErrorGradients = NULL;
        m_output              = new fml[m_numOutputDims * count];
        m_inputErrorGradients = new fml[m_numInputDims * count];
        m_maxCount = count;
    }
    m_curCount = count;

    if (m_numInputDims != m_numOutputDims)
        throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");
    for (u32 i = 0; i < count; i++)
    {
        for (u32 j = 0; j < numInputDims; j++)
        {
            m_output[i*numInputDims + j] = input[i*numInputDims + j] * m_p;
        }
    }
}

const fml* tDropoutLayerCPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_output;
}

void tDropoutLayerCPU::takeOutputErrorGradients(
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

    if (m_curCount == 0 || !m_output || !m_inputErrorGradients)
        throw eRuntimeError("What gives?");

    if (calculateInputErrorGradients)
    {
        if (m_numInputDims != m_numOutputDims)
            throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");
        for (u32 i = 0; i < inputCount; i++)
        {
            for (u32 j = 0; j < numInputDims; j++)
            {
                m_inputErrorGradients[i*numInputDims + j] = outputErrorGradients[i*numInputDims + j] * m_p;
            }
        }
    }
}

const fml* tDropoutLayerCPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_inputErrorGradients;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tDropoutLayerCPU* layer = new tDropoutLayerCPU();
    layer->unpack(in);
    return layer;
}

static u32 layerId = 9634374;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tDropoutLayerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}

void tDropoutLayerCPU::reset()
{
    // Always call the superclass impl no matter what.
    tDropoutLayerBase::reset();

    //
    // And if this subclass has its own things that need reseting, do it here.
    //
}

void tDropoutLayerCPU::pack(iWritable* out) const
{
    // Always call the superclass impl no matter what.
    tDropoutLayerBase::pack(out);

    //
    // Then, if this layer has its own things that need packed, do it here.
    //
}

void tDropoutLayerCPU::unpack(iReadable* in)
{
    // Always call the superclass impl no matter what.
    tDropoutLayerBase::unpack(in);

    //
    // Then, if this layer packed its own things, unpack them here.
    //

    //
    // Also, if there are other fields that need to be invalidated due to
    // unpacking other values, invalidate/reset everything here.
    //
}


}   // namespace ml
