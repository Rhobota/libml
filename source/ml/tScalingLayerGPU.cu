#include <ml/tScalingLayerGPU.h>

#include "cuda_stuff.ipp"


namespace ml
{


tScalingLayerGPU::tScalingLayerGPU()
    : tScalingLayerBase(),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL)
{
}

tScalingLayerGPU::tScalingLayerGPU(u32 numInputDims, u32 numOutputDims, fml scaleFactor)
    : tScalingLayerBase(numInputDims, numOutputDims, scaleFactor),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL)
{
}

tScalingLayerGPU::~tScalingLayerGPU()
{
    s_cudaFree(m_gpu_output);
    s_cudaFree(m_gpu_inputErrorGradients);
}

void tScalingLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count)
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
    if (m_numInputDims != m_numOutputDims)
        throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");
    for (u32 i = 0; i < count; i++)
    {
        for (u32 j = 0; j < numInputDims; j++)
        {
            m_gpu_output[i*numInputDims + j] = input[i*numInputDims + j] * m_scaleFactor;
        }
    }
}

const fml* tScalingLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numOutputDims;
    count = m_curCount;
    return m_gpu_output;
}

void tScalingLayerGPU::takeOutputErrorGradients(
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
    // from the given output error gradients. (In our case, this scaling layer doesn't learn anything.)
    //
    // <LEARNING CODE GOES HERE>

    // Below is the "backprop" step. Sometimes a layer doesn't need to back-propagate its error, thus
    // we check this condition and skip this work if it isn't needed.
    if (calculateInputErrorGradients)
    {
        if (m_numInputDims != m_numOutputDims)
            throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");
        for (u32 i = 0; i < inputCount; i++)
        {
            for (u32 j = 0; j < numInputDims; j++)
            {
                m_gpu_inputErrorGradients[i*numInputDims + j] = outputErrorGradients[i*numInputDims + j] * m_scaleFactor;
            }
        }
    }
}

const fml* tScalingLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_gpu_inputErrorGradients;
}

static
iLayer* s_newLayerFunc(iReadable* in)
{
    tScalingLayerGPU* layer = new tScalingLayerGPU();   // <-- Update this line for all new layer types.
    layer->unpack(in);
    return layer;
}

static u32 layerId = 432139;    // <-- Update this value inside all new layer types.
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);

u32 tScalingLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}

void tScalingLayerGPU::reset()
{
    // Always call the superclass impl no matter what.
    tScalingLayerBase::reset();

    //
    // And if this subclass has its own things that need reseting, do it here.
    //
}

void tScalingLayerGPU::pack(iWritable* out) const
{
    // Always call the superclass impl no matter what.
    tScalingLayerBase::pack(out);

    //
    // Then, if this layer has its own things that need packed, do it here.
    // Be sure to copy any GPU memory to host memory before you try to pack it!
    //
}

void tScalingLayerGPU::unpack(iReadable* in)
{
    // Always call the superclass impl no matter what.
    tScalingLayerBase::unpack(in);

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
