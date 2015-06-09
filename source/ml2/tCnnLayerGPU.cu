#include <ml2/tCnnLayerGPU.h>

#define ENABLE_DEVICE_FUNCTIONS
#define ENABLE_CONVOLVE_GPU
#include "common_nn.ipp"

#include <cassert>
#include <iostream>


namespace ml2
{


tCnnLayerGPU::tCnnLayerGPU()
    : tCnnLayerBase(),
      m_cublasContext(NULL),
      m_gpu_w(NULL),
      m_gpu_b(NULL),
      m_gpu_dw_accum(NULL),
      m_gpu_db_accum(NULL),
      m_gpu_A(NULL),
      m_gpu_a(NULL),
      m_gpu_dA(NULL),
      m_gpu_prev_da(NULL),
      m_gpu_vel(NULL),
      m_gpu_dw_accum_avg(NULL),
      m_gpu_uniqueKeys(NULL),
      m_gpu_columnSums(NULL),
      m_gpu_ones_vector(NULL)
{
    s_createCublasContext(m_cublasContext);
}


tCnnLayerGPU::tCnnLayerGPU(nLayerType type, nLayerWeightUpdateRule rule,
                           u32 inputRows, u32 inputCols, u32 inputComponents,
                           u32 kernelRows, u32 kernelCols,
                           u32 kernelStepY, u32 kernelStepX,
                           u32 numKernels,
                           algo::iLCG& lcg, fml randWeightMin, fml randWeightMax)
    : tCnnLayerBase(type, rule,
                    inputRows, inputCols, inputComponents,
                    kernelRows, kernelCols,
                    kernelStepY, kernelStepX,
                    numKernels,
                    lcg, randWeightMin, randWeightMax),
      m_cublasContext(NULL),
      m_gpu_w(NULL),
      m_gpu_b(NULL),
      m_gpu_dw_accum(NULL),
      m_gpu_db_accum(NULL),
      m_gpu_A(NULL),
      m_gpu_a(NULL),
      m_gpu_dA(NULL),
      m_gpu_prev_da(NULL),
      m_gpu_vel(NULL),
      m_gpu_dw_accum_avg(NULL),
      m_gpu_uniqueKeys(NULL),
      m_gpu_columnSums(NULL),
      m_gpu_ones_vector(NULL)
{
    s_createCublasContext(m_cublasContext);
    m_syncWeights_hostToDevice();
    m_initAccum();
}


tCnnLayerGPU::~tCnnLayerGPU()
{
    // The super d'tor is called automatically.

    m_finalize();
}


void tCnnLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t*)m_cublasContext;

    // TODO
}


const fml* tCnnLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_outputRows * m_outputCols * m_numKernels;
    count = m_curCount;
    return m_gpu_a;
}


void tCnnLayerGPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t*)m_cublasContext;

    // TODO
}


const fml* tCnnLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_inputRows * m_inputCols * m_inputComponents;
    count = m_curCount;
    return m_gpu_prev_da;
}


static
iLayer* s_newLayerFunc(iReadable* in)
{
    tCnnLayerGPU* layer = new tCnnLayerGPU();
    layer->unpack(in);
    return layer;
}


static u32 layerId = 9879879;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tCnnLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


void tCnnLayerGPU::reset()
{
    m_finalize();
    tCnnLayerBase::reset();
    s_createCublasContext(m_cublasContext);
    m_syncWeights_hostToDevice();
    m_initAccum();
}


void tCnnLayerGPU::pack(iWritable* out) const
{
    m_syncWeights_deviceToHost();
    tCnnLayerBase::pack(out);
}


void tCnnLayerGPU::unpack(iReadable* in)
{
    tCnnLayerBase::unpack(in);
    m_finalize();
    s_createCublasContext(m_cublasContext);
    m_syncWeights_hostToDevice();
    m_initAccum();
}


void tCnnLayerGPU::m_initAccum()
{
    s_cudaFree(m_gpu_dw_accum);
    s_cudaFree(m_gpu_db_accum);

    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    m_gpu_dw_accum = s_cudaMalloc(numWeights);
    thrust::device_ptr<fml> dw_accum(m_gpu_dw_accum);
    thrust::fill(dw_accum, dw_accum + numWeights, FML(0.0));

    u32 numBiases = m_numKernels;
    m_gpu_db_accum = s_cudaMalloc(numBiases);
    thrust::device_ptr<fml> db_accum(m_gpu_db_accum);
    thrust::fill(db_accum, db_accum + numBiases, FML(0.0));
}


void tCnnLayerGPU::m_finalize()
{
    s_cudaFree(m_gpu_w);
    s_cudaFree(m_gpu_b);
    s_cudaFree(m_gpu_dw_accum);
    s_cudaFree(m_gpu_db_accum);
    s_cudaFree(m_gpu_A);
    s_cudaFree(m_gpu_a);
    s_cudaFree(m_gpu_dA);
    s_cudaFree(m_gpu_prev_da);
    s_cudaFree(m_gpu_vel);
    s_cudaFree(m_gpu_dw_accum_avg);
    s_cudaFree(m_gpu_uniqueKeys);
    s_cudaFree(m_gpu_columnSums);
    s_cudaFree(m_gpu_ones_vector);

    s_destroyCublasContext(m_cublasContext);
}


void tCnnLayerGPU::m_syncWeights_deviceToHost() const
{
    if (!m_gpu_w || !m_gpu_b)
        throw eRuntimeError("Cannot sync weight from device to host because there are no device weights!");

    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    s_cudaCopyDeviceToHost(m_w, m_gpu_w, numWeights);

    u32 numBiases = m_numKernels;
    s_cudaCopyDeviceToHost(m_b, m_gpu_b, numBiases);
}


void tCnnLayerGPU::m_syncWeights_hostToDevice()
{
    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    s_cudaFree(m_gpu_w);
    m_gpu_w = s_cudaMalloc(numWeights);
    s_cudaCopyHostToDevice(m_gpu_w, m_w, numWeights);

    u32 numBiases = m_numKernels;
    s_cudaFree(m_gpu_b);
    m_gpu_b = s_cudaMalloc(numBiases);
    s_cudaCopyHostToDevice(m_gpu_b, m_b, numBiases);
}


}   // namespace ml2
