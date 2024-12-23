#include <ml/tPoolingLayerCPU.h>
#include <ml/pool2d/cpu_optimized.h>


namespace ml
{


tPoolingLayerCPU::tPoolingLayerCPU()
    : tPoolingLayerBase(),
      m_a(NULL),
      m_prev_da(NULL)
{
}

tPoolingLayerCPU::tPoolingLayerCPU(u32 inputRows, u32 inputCols, u32 inputComponents)
    : tPoolingLayerBase(inputRows, inputCols, inputComponents),
      m_a(NULL),
      m_prev_da(NULL)
{
}

tPoolingLayerCPU::tPoolingLayerCPU(u32 inputRows, u32 inputCols, u32 inputComponents,
                                   u32 poolRows, u32 poolCols)
    : tPoolingLayerBase(inputRows, inputCols, inputComponents,
                        poolRows, poolCols),
      m_a(NULL),
      m_prev_da(NULL)
{
}

tPoolingLayerCPU::~tPoolingLayerCPU()
{
    delete [] m_a;          m_a = NULL;
    delete [] m_prev_da;    m_prev_da = NULL;
}

void tPoolingLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count,
                                 bool isTrainMode, iLayer* prevLayer)
{
    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_inputRows*m_inputCols*m_inputComponents)
        throw eInvalidArgument("Unexpected numInputDims");

    if (count == 0)
        throw eInvalidArgument("count must be positive.");

    if (count > m_maxCount)
    {
        delete [] m_a;          m_a = NULL;
        delete [] m_prev_da;    m_prev_da = NULL;
        m_a       = new fml[(m_inputRows/m_poolRows) * (m_inputCols/m_poolCols) * m_inputComponents * count];
        m_prev_da = new fml[ m_inputRows             * m_inputCols              * m_inputComponents * count];
        m_maxCount = count;
    }
    m_curCount = count;

    pool2d::cpu_optimized::pool2d_multi_input(
            count,
            input,  m_inputRows,  m_inputCols,  m_inputComponents,
                    m_poolRows,  m_poolCols,
            m_a);
}

const fml* tPoolingLayerCPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = (m_inputRows/m_poolRows) * (m_inputCols/m_poolCols) * m_inputComponents;
    count = m_curCount;
    return m_a;
}

void tPoolingLayerCPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    if (!outputErrorGradients)
        throw eInvalidArgument("outputErrorGradients may not be null!");

    if (numOutputDims != (m_inputRows/m_poolRows) * (m_inputCols/m_poolCols) * m_inputComponents)
        throw eInvalidArgument("Unexpected numOutputDims");

    if (outputCount != m_curCount)
        throw eInvalidArgument("Unexpected outputCount");

    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_inputRows*m_inputCols*m_inputComponents)
        throw eInvalidArgument("Unexpected numInputDims");

    if (inputCount != m_curCount)
        throw eInvalidArgument("Unexpected inputCount");

    if (m_curCount == 0 || !m_a || !m_prev_da)
        throw eRuntimeError("What gives?");

    if (calculateInputErrorGradients)
    {
        pool2d::cpu_optimized::un_pool2d_multi_input(
                inputCount,
                input,  m_inputRows,  m_inputCols,  m_inputComponents,
                        m_poolRows,  m_poolCols,
                outputErrorGradients,
                m_prev_da);
    }
}

const fml* tPoolingLayerCPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_inputRows*m_inputCols*m_inputComponents;
    count = m_curCount;
    return m_prev_da;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tPoolingLayerCPU* layer = new tPoolingLayerCPU();
    layer->unpack(in);
    return layer;
}

static u32 layerId = 732192;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tPoolingLayerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


}   // namespace ml
