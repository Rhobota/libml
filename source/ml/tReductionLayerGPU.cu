#include <ml/tReductionLayerGPU.h>

#include "cuda_stuff.ipp"


namespace ml
{


class tColumnIndexFunc : public thrust::unary_function<u32,u32>
{
    public:

        tColumnIndexFunc(u32 numRows)
            : m_numRows(numRows) { }

        __host__ __device__
        u32 operator()(u32 index)
        {
            return (index / m_numRows);
        }

    private:

        u32 m_numRows;
};


class tFillDuplicateValues
{
    public:

        tFillDuplicateValues(const fml* values, u32 numDuplicates)
            : m_values(values), m_numDuplicates(numDuplicates) { }

        __host__ __device__
        fml operator()(const ssize_t& index)
        {
            return m_values[(index / m_numDuplicates)];
        }

    private:

        const fml* m_values;
        u32  m_numDuplicates;
};


tReductionLayerGPU::tReductionLayerGPU()
    : tReductionLayerBase(),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL)
{
}

tReductionLayerGPU::tReductionLayerGPU(u32 numInputDims, u32 numOutputDims)
    : tReductionLayerBase(numInputDims, numOutputDims),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL)
{
}

tReductionLayerGPU::~tReductionLayerGPU()
{
    s_cudaFree(m_gpu_output);
    s_cudaFree(m_gpu_inputErrorGradients);
}

void tReductionLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count,
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
        s_cudaFree(m_gpu_output);
        s_cudaFree(m_gpu_inputErrorGradients);
        m_gpu_output              = s_cudaMalloc(m_numOutputDims * count);
        m_gpu_inputErrorGradients = s_cudaMalloc(m_numInputDims * count);
        m_maxCount = count;
    }
    m_curCount = count;

    // Below is the actual work that this layer is designed to do. We must use
    // the given input to calculate this layer's output.
    if (m_numOutputDims != 1)
        throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have output dimensionality not equal one.");
    thrust::device_ptr<const fml> inputItr(input);
    thrust::device_ptr<      fml> outputItr(m_gpu_output);
    tColumnIndexFunc colIndexFunc(numInputDims);
    thrust::reduce_by_key(
        thrust::make_transform_iterator(
            thrust::make_counting_iterator((u32)0),
            colIndexFunc),
        thrust::make_transform_iterator(
            thrust::make_counting_iterator((u32)0),
            colIndexFunc) + numInputDims*count,
        inputItr,
        thrust::make_discard_iterator(),
        outputItr);
}

const fml* tReductionLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_gpu_output;
}

void tReductionLayerGPU::takeOutputErrorGradients(
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

    if (m_curCount == 0 || !m_gpu_output || !m_gpu_inputErrorGradients)
        throw eRuntimeError("What gives?");

    // Note: If this layer was a "learning" layer, there would be some work to do here to "learn"
    // from the given output error gradients. (In our case, this reduction layer doesn't learn anything.)
    //
    // <LEARNING CODE GOES HERE>

    // Below is the "backprop" step. Sometimes a layer doesn't need to back-propagate its error, thus
    // we check this condition and skip this work if it isn't needed.
    if (calculateInputErrorGradients)
    {
        if (m_numOutputDims != 1)
            throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have output dimensionality not equal one.");
        thrust::device_ptr<fml> itr(m_gpu_inputErrorGradients);
        tFillDuplicateValues func(outputErrorGradients, numInputDims);
        thrust::tabulate(itr, itr + numInputDims*inputCount, func);
    }
}

const fml* tReductionLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_gpu_inputErrorGradients;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tReductionLayerGPU* layer = new tReductionLayerGPU();   // <-- Update this line for all new layer types.
    layer->unpack(in);
    return layer;
}

static u32 layerId = 86713257;    // <-- Update this value inside all new layer types.
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tReductionLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}

void tReductionLayerGPU::reset()
{
    // Always call the superclass impl no matter what.
    tReductionLayerBase::reset();

    //
    // And if this subclass has its own things that need reseting, do it here.
    //
}

void tReductionLayerGPU::pack(iWritable* out) const
{
    // Always call the superclass impl no matter what.
    tReductionLayerBase::pack(out);

    //
    // Then, if this layer has its own things that need packed, do it here.
    // Be sure to copy any GPU memory to host memory before you try to pack it!
    //
}

void tReductionLayerGPU::unpack(iReadable* in)
{
    // Always call the superclass impl no matter what.
    tReductionLayerBase::unpack(in);

    //
    // Then, if this layer packed its own things, unpack them here.
    // Be sure to copy the memory to the GPU if applicable.
    //

    //
    // Also, if there are other fields that need to be invalidated due to
    // unpacking other values, invalidate/reset everything here.
    //
}


}   // namespace ml
