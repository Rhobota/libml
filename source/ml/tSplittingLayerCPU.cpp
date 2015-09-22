#include <ml/tSplittingLayerCPU.h>


namespace ml
{


tSplittingLayerCPU::tSplittingLayerCPU()
    : tSplittingLayerBase(),
      m_a(NULL),
      m_prev_da(NULL)
{
}

tSplittingLayerCPU::tSplittingLayerCPU(u32 numInputDims, u32 numOutputDims)
    : tSplittingLayerBase(numInputDims, numOutputDims),
      m_a(NULL),
      m_prev_da(NULL)
{
}

tSplittingLayerCPU::~tSplittingLayerCPU()
{
    delete [] m_a;          m_a = NULL;
    delete [] m_prev_da;    m_prev_da = NULL;
    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        delete [] rec.inputPtr;
        rec.inputPtr = NULL;
        delete [] rec.outputErrorPtr;
        rec.outputErrorPtr = NULL;
        delete rec.layer;
        rec.layer = NULL;
    }
    m_layerRecords.clear();
}

void tSplittingLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count,
                                   bool isTrainMode, iLayer* prevLayer)
{
    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims");

    u32 sum = 0;
    for (size_t i = 0; i < m_layerRecords.size(); i++)
        sum += m_layerRecords[i].numInputDims;
    if (sum != m_numInputDims)
        throw eRuntimeError("The sub-layers' input dims don't add up to this layer's input dims.");

    sum = 0;
    for (size_t i = 0; i < m_layerRecords.size(); i++)
        sum += m_layerRecords[i].numOutputDims;
    if (sum != m_numOutputDims)
        throw eRuntimeError("The sub-layers' output dims don't add up to this layer's output dims.");

    if (count == 0)
        throw eInvalidArgument("count must be positive.");

    if (count > m_maxCount)
    {
        delete [] m_a;          m_a = NULL;
        delete [] m_prev_da;    m_prev_da = NULL;
        m_a       = new fml[m_numOutputDims * count];
        m_prev_da = new fml[m_numInputDims * count];
        m_maxCount = count;
        for (size_t i = 0; i < m_layerRecords.size(); i++)
        {
            tLayerRecord& rec = m_layerRecords[i];
            delete [] rec.inputPtr;
            rec.inputPtr = new fml[rec.numInputDims * count];
            delete [] rec.outputErrorPtr;
            rec.outputErrorPtr = new fml[rec.numOutputDims * count];
        }
    }
    m_curCount = count;

    for (u32 c = 0; c < count; c++)
    {
        for (size_t i = 0; i < m_layerRecords.size(); i++)
        {
            tLayerRecord& rec = m_layerRecords[i];
            u32 d = rec.numInputDims;
            fml* layerIn = rec.inputPtr + c * d;
            for (u32 j = 0; j < d; j++)
            {
                *layerIn++ = *input++;
            }
        }
    }

    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        rec.layer->takeInput(rec.inputPtr, rec.numInputDims, count,
                             isTrainMode, prevLayer);
    }

    fml* output = m_a;
    for (u32 c = 0; c < count; c++)
    {
        for (size_t i = 0; i < m_layerRecords.size(); i++)
        {
            tLayerRecord& rec = m_layerRecords[i];
            u32 d = rec.numOutputDims;
            u32 d2 = 0, c2 = 0;
            const fml* layerOut = rec.layer->getOutput(d2, c2) + c * d;
            if (d2 != d)
                throw eRuntimeError("Unexpected numOutputDims from sublayer.");
            if (c2 != count)
                throw eRuntimeError("Unexpected count from sublayer.");
            for (u32 j = 0; j < d; j++)
            {
                 *output++ = *layerOut++;
            }
        }
    }
}

const fml* tSplittingLayerCPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_a;
}

void tSplittingLayerCPU::takeOutputErrorGradients(
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

    if (m_curCount == 0 || !m_a || !m_prev_da)
        throw eRuntimeError("What gives?");

    u32 sum = 0;
    for (size_t i = 0; i < m_layerRecords.size(); i++)
        sum += m_layerRecords[i].numInputDims;
    if (sum != m_numInputDims)
        throw eRuntimeError("The sub-layers' input dims don't add up to this layer's input dims.");

    sum = 0;
    for (size_t i = 0; i < m_layerRecords.size(); i++)
        sum += m_layerRecords[i].numOutputDims;
    if (sum != m_numOutputDims)
        throw eRuntimeError("The sub-layers' output dims don't add up to this layer's output dims.");

    for (u32 c = 0; c < inputCount; c++)
    {
        for (size_t i = 0; i < m_layerRecords.size(); i++)
        {
            tLayerRecord& rec = m_layerRecords[i];
            u32 d = rec.numOutputDims;
            fml* layerOut = rec.outputErrorPtr + c * d;
            for (u32 j = 0; j < d; j++)
            {
                *layerOut++ = *outputErrorGradients++;
            }
        }
    }

    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        rec.layer->takeOutputErrorGradients(rec.outputErrorPtr, rec.numOutputDims, outputCount,
                                            rec.inputPtr, rec.numInputDims, inputCount,
                                            calculateInputErrorGradients);
    }

    if (calculateInputErrorGradients)
    {
        fml* inError = m_prev_da;
        for (u32 c = 0; c < inputCount; c++)
        {
            for (size_t i = 0; i < m_layerRecords.size(); i++)
            {
                tLayerRecord& rec = m_layerRecords[i];
                u32 d = rec.numInputDims;
                u32 d2 = 0, c2 = 0;
                const fml* layerInError = rec.layer->getInputErrorGradients(d2, c2) + c * d;
                if (d2 != d)
                    throw eRuntimeError("Unexpected numInputDims from sublayer.");
                if (c2 != inputCount)
                    throw eRuntimeError("Unexpected inputCount from sublayer.");
                for (u32 j = 0; j < d; j++)
                {
                     *inError++ = *layerInError++;
                }
            }
        }
    }
}

const fml* tSplittingLayerCPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_prev_da;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tSplittingLayerCPU* layer = new tSplittingLayerCPU();
    layer->unpack(in);
    return layer;
}

static u32 layerId = 832192;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tSplittingLayerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


}   // namespace ml
