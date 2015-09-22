#include <ml/tPoolingLayerGPU.h>
#include <ml/pool2d/gpu.h>

#include "cuda_stuff.ipp"


namespace ml
{


tPoolingLayerGPU::tPoolingLayerGPU()
    : tPoolingLayerBase(),
      m_gpu_a(NULL),
      m_gpu_prev_da(NULL)
{
}

tPoolingLayerGPU::tPoolingLayerGPU(u32 inputRows, u32 inputCols, u32 inputComponents)
    : tPoolingLayerBase(inputRows, inputCols, inputComponents),
      m_gpu_a(NULL),
      m_gpu_prev_da(NULL)
{
}

tPoolingLayerGPU::tPoolingLayerGPU(u32 inputRows, u32 inputCols, u32 inputComponents,
                                   u32 poolRows, u32 poolCols)
    : tPoolingLayerBase(inputRows, inputCols, inputComponents,
                        poolRows, poolCols),
      m_gpu_a(NULL),
      m_gpu_prev_da(NULL)
{
}

tPoolingLayerGPU::~tPoolingLayerGPU()
{
    s_cudaFree(m_gpu_a);
    s_cudaFree(m_gpu_prev_da);
}

void tPoolingLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count,
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
        s_cudaFree(m_gpu_a);
        s_cudaFree(m_gpu_prev_da);
        m_gpu_a       = s_cudaMalloc((m_inputRows/m_poolRows) * (m_inputCols/m_poolCols) * m_inputComponents * count);
        m_gpu_prev_da = s_cudaMalloc( m_inputRows             * m_inputCols              * m_inputComponents * count);
        m_maxCount = count;
    }
    m_curCount = count;

    pool2d::gpu::pool2d_multi_input(
            count,
            input,  m_inputRows,  m_inputCols,  m_inputComponents,
                    m_poolRows,  m_poolCols,
            m_gpu_a);
}

const fml* tPoolingLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = (m_inputRows/m_poolRows) * (m_inputCols/m_poolCols) * m_inputComponents;
    count = m_curCount;
    return m_gpu_a;
}

void tPoolingLayerGPU::takeOutputErrorGradients(
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

    if (m_curCount == 0 || !m_gpu_a || !m_gpu_prev_da)
        throw eRuntimeError("What gives?");

    if (calculateInputErrorGradients)
    {
        pool2d::gpu::un_pool2d_multi_input(
                inputCount,
                input,  m_inputRows,  m_inputCols,  m_inputComponents,
                        m_poolRows,  m_poolCols,
                outputErrorGradients,
                m_gpu_prev_da);
    }
}

const fml* tPoolingLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_inputRows*m_inputCols*m_inputComponents;
    count = m_curCount;
    return m_gpu_prev_da;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tPoolingLayerGPU* layer = new tPoolingLayerGPU();
    layer->unpack(in);
    return layer;
}

static u32 layerId = 732193;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tPoolingLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


}   // namespace ml
