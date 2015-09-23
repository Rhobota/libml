#include <ml/tDropoutLayerCPU.h>


namespace ml
{


static
fml s_bernoulli(algo::iLCG& lcg, fml p)
{
    u64 cutoff = (u64) ceil(((f64)(lcg.randMax() + 1)) * ((f64)p));
    return (lcg.next() < cutoff) ? FML(1.0) : FML(0.0);
}


tDropoutLayerCPU::tDropoutLayerCPU()
    : tDropoutLayerBase(),
      m_output(NULL),
      m_inputErrorGradients(NULL),
      m_lcg(),
      m_dropMask(NULL)
{
}

tDropoutLayerCPU::tDropoutLayerCPU(u32 numInputDims, u32 numOutputDims, u64 rndSeed, fml p)
    : tDropoutLayerBase(numInputDims, numOutputDims, p),
      m_output(NULL),
      m_inputErrorGradients(NULL),
      m_lcg(rndSeed+1),
      m_dropMask(NULL)
{
}

tDropoutLayerCPU::~tDropoutLayerCPU()
{
    delete [] m_output;                 m_output = NULL;
    delete [] m_inputErrorGradients;    m_inputErrorGradients = NULL;
    delete [] m_dropMask;               m_dropMask = NULL;
}

void tDropoutLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count,
                                 bool isTrainMode, iLayer* prevLayer)
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
        delete [] m_dropMask;               m_dropMask = NULL;
        m_output              = new fml[m_numOutputDims * count];
        m_inputErrorGradients = new fml[m_numInputDims * count];
        m_dropMask            = new fml[m_numInputDims * count];
        m_maxCount = count;
    }
    m_curCount = count;

    if (m_numInputDims != m_numOutputDims)
        throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");

    m_trainMode = isTrainMode;

    if (m_trainMode)   // <-- train mode
    {
        for (u32 i = 0; i < count; i++)
        {
            for (u32 j = 0; j < numInputDims; j++)
            {
                fml drop = s_bernoulli(m_lcg, m_p);
                m_dropMask[i*numInputDims + j] = drop;
                m_output[i*numInputDims + j] = input[i*numInputDims + j] * drop;
            }
        }
    }

    else               // <-- test mode
    {
        for (u32 i = 0; i < count; i++)
        {
            for (u32 j = 0; j < numInputDims; j++)
            {
                m_output[i*numInputDims + j] = input[i*numInputDims + j] * m_p;
            }
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

    if (m_curCount == 0 || !m_output || !m_inputErrorGradients || !m_dropMask)
        throw eRuntimeError("What gives?");

    if (calculateInputErrorGradients)
    {
        if (m_numInputDims != m_numOutputDims)
            throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");

        if (m_trainMode)
        {
            for (u32 i = 0; i < inputCount; i++)
            {
                for (u32 j = 0; j < numInputDims; j++)
                {
                    m_inputErrorGradients[i*numInputDims + j] = outputErrorGradients[i*numInputDims + j] * m_dropMask[i*numInputDims + j];
                }
            }
        }

        else
        {
            throw eRuntimeError("Why are you trying to train this layer while it's not in training mode!? Don't do that.");
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
    tDropoutLayerBase::reset();
}

void tDropoutLayerCPU::pack(iWritable* out) const
{
    tDropoutLayerBase::pack(out);
}

void tDropoutLayerCPU::unpack(iReadable* in)
{
    tDropoutLayerBase::unpack(in);
}


}   // namespace ml
