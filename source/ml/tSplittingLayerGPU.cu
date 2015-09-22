#include <ml/tSplittingLayerGPU.h>

#include "cuda_stuff.ipp"


namespace ml
{


class tSublayerFillerFunc
{
    public:

        tSublayerFillerFunc(const fml* input, u32 outerDims, u32 innerDims)
            : m_input(input), m_outerDims(outerDims), m_innerDims(innerDims)
        { }

        __host__ __device__
        fml operator()(const ssize_t& index)
        {
            size_t count = index / m_innerDims;
            size_t offset = index % m_innerDims;
            return m_input[count*m_outerDims + offset];
        }

    private:

        const fml* m_input;
        u32 m_outerDims;
        u32 m_innerDims;
};


class tConvertIndexFunc : public thrust::unary_function<size_t, size_t>
{
    public:

        tConvertIndexFunc(u32 outerDims, u32 innerDims)
            : m_outerDims(outerDims), m_innerDims(innerDims)
        { }

        __host__ __device__
        size_t operator()(size_t index)
        {
            size_t count = index / m_innerDims;
            size_t offset = index % m_innerDims;
            return count*m_outerDims + offset;
        }

    private:

        u32 m_outerDims;
        u32 m_innerDims;
};


tSplittingLayerGPU::tSplittingLayerGPU()
    : tSplittingLayerBase(),
      m_gpu_a(NULL),
      m_gpu_prev_da(NULL)
{
}

tSplittingLayerGPU::tSplittingLayerGPU(u32 numInputDims, u32 numOutputDims)
    : tSplittingLayerBase(numInputDims, numOutputDims),
      m_gpu_a(NULL),
      m_gpu_prev_da(NULL)
{
}

tSplittingLayerGPU::~tSplittingLayerGPU()
{
    s_cudaFree(m_gpu_a);
    s_cudaFree(m_gpu_prev_da);
    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        s_cudaFree(rec.inputPtr);
        rec.inputPtr = NULL;
        s_cudaFree(rec.outputErrorPtr);
        rec.outputErrorPtr = NULL;
        delete rec.layer;
        rec.layer = NULL;
    }
    m_layerRecords.clear();
}

void tSplittingLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count,
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
        s_cudaFree(m_gpu_a);
        s_cudaFree(m_gpu_prev_da);
        m_gpu_a       = s_cudaMalloc(m_numOutputDims * count);
        m_gpu_prev_da = s_cudaMalloc(m_numInputDims * count);
        m_maxCount = count;
        for (size_t i = 0; i < m_layerRecords.size(); i++)
        {
            tLayerRecord& rec = m_layerRecords[i];
            s_cudaFree(rec.inputPtr);
            rec.inputPtr = s_cudaMalloc(rec.numInputDims * count);
            s_cudaFree(rec.outputErrorPtr);
            rec.outputErrorPtr = s_cudaMalloc(rec.numOutputDims * count);
        }
    }
    m_curCount = count;

    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        thrust::device_ptr<fml> layerIn(rec.inputPtr);
        tSublayerFillerFunc func(input, numInputDims, rec.numInputDims);
        thrust::tabulate(layerIn, layerIn + rec.numInputDims * count, func);
        input += rec.numInputDims;
    }

    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        rec.layer->takeInput(rec.inputPtr, rec.numInputDims, count,
                             isTrainMode, prevLayer);
    }

    thrust::device_ptr<fml> output(m_gpu_a);
    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        u32 d = rec.numOutputDims;
        u32 d2 = 0, c2 = 0;
        thrust::device_ptr<const fml> layerOut(rec.layer->getOutput(d2, c2));
        if (d2 != d)
            throw eRuntimeError("Unexpected numOutputDims from sublayer.");
        if (c2 != count)
            throw eRuntimeError("Unexpected count from sublayer.");
        thrust::scatter(layerOut, layerOut + d * count,
                        thrust::make_transform_iterator(thrust::make_counting_iterator((size_t)0),
                                                        tConvertIndexFunc(m_numOutputDims, d)),
                        output);
        output += d;
    }
}

const fml* tSplittingLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_gpu_a;
}

void tSplittingLayerGPU::takeOutputErrorGradients(
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

    if (m_curCount == 0 || !m_gpu_a || !m_gpu_prev_da)
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

    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        tLayerRecord& rec = m_layerRecords[i];
        thrust::device_ptr<fml> layerOut(rec.outputErrorPtr);
        tSublayerFillerFunc func(outputErrorGradients, numOutputDims, rec.numOutputDims);
        thrust::tabulate(layerOut, layerOut + rec.numOutputDims * inputCount, func);
        outputErrorGradients += rec.numOutputDims;
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
        thrust::device_ptr<fml> inError(m_gpu_prev_da);
        for (size_t i = 0; i < m_layerRecords.size(); i++)
        {
            tLayerRecord& rec = m_layerRecords[i];
            u32 d = rec.numInputDims;
            u32 d2 = 0, c2 = 0;
            thrust::device_ptr<const fml> layerInError(rec.layer->getInputErrorGradients(d2, c2));
            if (d2 != d)
                throw eRuntimeError("Unexpected numInputDims from sublayer.");
            if (c2 != inputCount)
                throw eRuntimeError("Unexpected inputCount from sublayer.");
            thrust::scatter(layerInError, layerInError + d * inputCount,
                            thrust::make_transform_iterator(thrust::make_counting_iterator((size_t)0),
                                                            tConvertIndexFunc(m_numInputDims, d)),
                            inError);
            inError += d;
        }
    }
}

const fml* tSplittingLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_gpu_prev_da;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tSplittingLayerGPU* layer = new tSplittingLayerGPU();
    layer->unpack(in);
    return layer;
}

static u32 layerId = 832193;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tSplittingLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


}   // namespace ml
