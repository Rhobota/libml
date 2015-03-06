#include <ml2/tAnnLayerGPU.h>

#include <cuda.h>
#include <cublas_v2.h>

#include <cassert>
#include <iostream>


#define cuda_assert(expression) \
    do { \
        cudaError_t err; \
        if ((err = (expression)) != cudaSuccess) \
        { \
            std::cout << "Cuda error: " << cudaGetErrorString(err) << std::endl; \
            assert(false); \
        } \
    } while (false)


#define cublas_assert(expression) \
    do { \
        if ((expression) != CUBLAS_STATUS_SUCCESS) \
        { \
            std::cout << "cuBLAS error!" << std::endl; \
            assert(false); \
        } \
    } while (false)


namespace ml2
{


static
void s_cudaFree(fml*& buf)
{
    if (buf)
    {
        cuda_assert( cudaFree(buf) );
        buf = NULL;
    }
}


class tExpFunc
{
    public:

        fml operator()(fml val) const { return std::min(std::exp(val), FML(1e30)); }
};


class tLogisticFunc
{
    public:

        fml operator()(fml val) const { return logistic_function(val); }
};


class tDirLogisticFunc
{
    public:

        fml operator()(fml val) const { return derivative_of_logistic_function(val); }
};


class tHyperbolicFunc
{
    public:

        fml operator()(fml val) const { return hyperbolic_function(val); }
};


class tDirHyperbolicFunc
{
    public:

        fml operator()(fml val) const { return derivative_of_hyperbolic_function(val); }
};


class t_RMSPROP_update
{
    public:

        fml operator()(fml accum, fml accum_avg) const
        {
            return (accum_avg > FML(0.0)) ? (accum / std::sqrt(accum_avg)) : FML(0.0);
        }
};


tAnnLayerGPU::tAnnLayerGPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                           u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                           fml randWeightMin, fml randWeightMax)
    : tAnnLayerBase(type, rule, numInputDims, numNeurons, lcg,
                    randWeightMin, randWeightMax),
      m_gpu_w(NULL),
      m_gpu_b(NULL),
      m_gpu_dw_accum(NULL),
      m_gpu_db_accum(NULL),
      m_gpu_A(NULL),
      m_gpu_a(NULL),
      m_gpu_dA(NULL),
      m_gpu_prev_da(NULL),
      m_gpu_vel(NULL),
      m_gpu_dw_accum_avg(NULL)
{
}


tAnnLayerGPU::tAnnLayerGPU(iReadable* in)
    : tAnnLayerBase(in),
      m_gpu_w(NULL),
      m_gpu_b(NULL),
      m_gpu_dw_accum(NULL),
      m_gpu_db_accum(NULL),
      m_gpu_A(NULL),
      m_gpu_a(NULL),
      m_gpu_dA(NULL),
      m_gpu_prev_da(NULL),
      m_gpu_vel(NULL),
      m_gpu_dw_accum_avg(NULL)
{
}


tAnnLayerGPU::~tAnnLayerGPU()
{
    // The super d'tor are called automatically.

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
}


void tAnnLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    // TODO
}


const fml* tAnnLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    // TODO
    return NULL;
}


void tAnnLayerGPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    // TODO
}


const fml* tAnnLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    // TODO
    return NULL;
}


static
iLayer* s_newLayerFunc(iReadable* in)
{
    return new tAnnLayerGPU(in);
}


static u32 layerId = 78879;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tAnnLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


void tAnnLayerGPU::m_syncWeights_deviceToHost()
{
    // TODO
}


void tAnnLayerGPU::m_syncWeights_hostToDevice()
{
    // TODO
}


}   // namespace ml2
