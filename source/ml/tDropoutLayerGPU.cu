#include <ml/tDropoutLayerGPU.h>

#include "cuda_stuff.ipp"


namespace ml
{


class tThreshFunc
{
    public:

        tThreshFunc(fml thresh)
            : m_thresh(thresh)
        { }

        __host__ __device__
        fml operator()(fml val)
        {
            return (val <= m_thresh) ? FML(1.0) : FML(0.0);
        }

    private:

        fml m_thresh;
};


class tMultFunc
{
    public:

        tMultFunc()
        { }

        __host__ __device__
        fml operator()(fml a, fml b)
        {
            return a * b;
        }

    private:

};


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
      m_gpu_inputErrorGradients(NULL),
      m_curandGen(NULL),
      m_gpu_dropMask(NULL)
{
}

tDropoutLayerGPU::tDropoutLayerGPU(u32 numInputDims, u32 numOutputDims, u64 rndSeed, fml p)
    : tDropoutLayerBase(numInputDims, numOutputDims, rndSeed, p),
      m_gpu_output(NULL),
      m_gpu_inputErrorGradients(NULL),
      m_curandGen(NULL),
      m_gpu_dropMask(NULL)
{
    s_createCurandGenerator(m_curandGen, rndSeed+1);
}

tDropoutLayerGPU::~tDropoutLayerGPU()
{
    s_cudaFree(m_gpu_output);
    s_cudaFree(m_gpu_inputErrorGradients);

    s_destroyCurandGenerator(m_curandGen);
    s_cudaFree(m_gpu_dropMask);
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
        s_cudaFree(m_gpu_dropMask);
        m_gpu_output              = s_cudaMalloc(m_numOutputDims * count);
        m_gpu_inputErrorGradients = s_cudaMalloc(m_numInputDims * count);
        m_gpu_dropMask            = s_cudaMalloc(m_numInputDims * count);
        m_maxCount = count;
    }
    m_curCount = count;

    if (m_numInputDims != m_numOutputDims)
        throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");

    m_trainMode = isTrainMode;

    if (m_trainMode)   // <-- train mode
    {
        thrust::device_ptr<const fml> inputItr(input);
        thrust::device_ptr<      fml> dropMask(m_gpu_dropMask);
        thrust::device_ptr<      fml> outputItr(m_gpu_output);

        s_curandGenerateUniform(m_curandGen, m_gpu_dropMask, numInputDims*count);   // <-- generates in (0.0, 1.0]
        tThreshFunc threshFunc(m_p);
        thrust::transform(dropMask, dropMask + numInputDims*count, dropMask, threshFunc);

        tMultFunc multFunc;
        thrust::transform(inputItr, inputItr + numInputDims*count, dropMask, outputItr, multFunc);
    }

    else               // <-- test mode
    {
        thrust::device_ptr<const fml> inputItr(input);
        thrust::device_ptr<      fml> outputItr(m_gpu_output);
        tScalarMultFunc func(m_p);
        thrust::transform(inputItr, inputItr + numInputDims*count, outputItr, func);
    }
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

    if (m_curCount == 0 || !m_gpu_output || !m_gpu_inputErrorGradients || !m_gpu_dropMask)
        throw eRuntimeError("What gives?");

    if (calculateInputErrorGradients)
    {
        if (m_numInputDims != m_numOutputDims)
            throw eInvalidArgument("Oops. It makes no sense for this kind of layer to have different input and output dimensionality.");

        if (m_trainMode)
        {
            thrust::device_ptr<const fml> inputItr(outputErrorGradients);
            thrust::device_ptr<      fml> dropMask(m_gpu_dropMask);
            thrust::device_ptr<      fml> outputItr(m_gpu_inputErrorGradients);
            tMultFunc multFunc;
            thrust::transform(inputItr, inputItr + numInputDims*inputCount, dropMask, outputItr, multFunc);
        }

        else
        {
            throw eRuntimeError("Why are you trying to train this layer while it's not in training mode!? Don't do that.");
        }
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
    tDropoutLayerBase::reset();
    s_destroyCurandGenerator(m_curandGen);
    s_createCurandGenerator(m_curandGen, m_rndSeed+1);
}

void tDropoutLayerGPU::pack(iWritable* out) const
{
    tDropoutLayerBase::pack(out);
}

void tDropoutLayerGPU::unpack(iReadable* in)
{
    tDropoutLayerBase::unpack(in);
    reset();
}


}   // namespace ml
