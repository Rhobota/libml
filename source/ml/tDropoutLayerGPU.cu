#include <ml/tDropoutLayerGPU.h>

#include "cuda_stuff.ipp"


namespace ml
{


class tScalarMultFunc
{
    public:

        tScalarMultFunc(fml scaleFactor)
            : m_scaleFactor(scaleFactor)
        { }

        __host__ __device__
        fml operator()(fml val)
        {
            return val * m_scaleFactor;
        }

    private:

        fml m_scaleFactor;
};


tDropoutLayerGPU::tDropoutLayerGPU()
    : tDropoutLayerBase(),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL)
{
}

tDropoutLayerGPU::tDropoutLayerGPU(u32 numInputDims, u32 numOutputDims, u32 rndSeed, fml p)
    : tDropoutLayerBase(numInputDims, numOutputDims, p),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL)
{
}

tDropoutLayerGPU::~tDropoutLayerGPU()
{
    s_cudaFree(m_gpu_output);
    s_cudaFree(m_gpu_inputErrorGradients);
}

void tDropoutLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count,
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

    if (m_numInputDims != m_numOutputDims)
        throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");
    thrust::device_ptr<const fml> inputItr(input);
    thrust::device_ptr<      fml> outputItr(m_gpu_output);
    tScalarMultFunc func(m_scaleFactor);
    thrust::transform(inputItr, inputItr + numInputDims*count, outputItr, func);
}

const fml* tDropoutLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_gpu_output;
}

void tDropoutLayerGPU::takeOutputErrorGradients(
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

    if (calculateInputErrorGradients)
    {
        if (m_numInputDims != m_numOutputDims)
            throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");
        thrust::device_ptr<const fml> inputItr(outputErrorGradients);
        thrust::device_ptr<      fml> outputItr(m_gpu_inputErrorGradients);
        tScalarMultFunc func(m_scaleFactor);
        thrust::transform(inputItr, inputItr + numInputDims*inputCount, outputItr, func);
    }
}

const fml* tDropoutLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_gpu_inputErrorGradients;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tDropoutLayerGPU* layer = new tDropoutLayerGPU();
    layer->unpack(in);
    return layer;
}

static u32 layerId = 9634375;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tDropoutLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}

void tDropoutLayerGPU::reset()
{
    // Always call the superclass impl no matter what.
    tDropoutLayerBase::reset();

    //
    // And if this subclass has its own things that need reseting, do it here.
    //
}

void tDropoutLayerGPU::pack(iWritable* out) const
{
    // Always call the superclass impl no matter what.
    tDropoutLayerBase::pack(out);

    //
    // Then, if this layer has its own things that need packed, do it here.
    // Be sure to copy any GPU memory to host memory before you try to pack it!
    //
}

void tDropoutLayerGPU::unpack(iReadable* in)
{
    // Always call the superclass impl no matter what.
    tDropoutLayerBase::unpack(in);

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
