#include <ml2/tAnnLayerGPU.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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


#define cublas_assert(expression, what) \
    do { \
        if ((expression) != CUBLAS_STATUS_SUCCESS) \
        { \
            std::cout << "cuBLAS error! " << what << std::endl; \
            assert(false); \
        } \
    } while (false)


namespace ml2
{


#define ENABLE_DEVICE_FUNCTIONS
#include "common.ipp"


static
void s_cudaFree(fml*& buf)
{
    if (buf)
    {
        cuda_assert( cudaFree(buf) );
        buf = NULL;
    }
}


static
fml* s_cudaMalloc(u32 size)
{
    fml* ptr = NULL;
    cuda_assert( cudaMalloc((void**)(&ptr), size*sizeof(fml)) );
    return ptr;
}


static
void s_createCublasContext(void*& ptr)
{
    cublasHandle_t* cublasHandle = new cublasHandle_t;
    cublas_assert( cublasCreate(cublasHandle), "s_createCublasContext" );
    ptr = cublasHandle;
}


static
void s_destroyCublasContext(void*& ptr)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t*)ptr;
    cublas_assert( cublasDestroy(*cublasHandle), "s_destroyCublasContext" );
    delete cublasHandle;
    ptr = NULL;
}


class tFillColumnsWithFunc
{
    public:

        tFillColumnsWithFunc(fml* vect, u32 vectSize)
            : m_vect(vect), m_vectSize(vectSize) { }

        __device__
        fml operator()(const ssize_t& index)
        {
            return m_vect[(index % m_vectSize)];
        }

    private:

        fml* m_vect;
        u32  m_vectSize;
};


class tColumnIndexFunc : public thrust::unary_function<u32,u32>
{
    public:

        tColumnIndexFunc(u32 numRows)
            : m_numRows(numRows) { }

        __device__
        u32 operator()(u32 index)
        {
            return (index / m_numRows);
        }

    private:

        u32 m_numRows;
};


class tDivInputColsByVectorValues
{
    public:

        tDivInputColsByVectorValues(fml* input, fml* vect, u32 numInputRows)
            : m_input(input), m_vect(vect), m_numInputRows(numInputRows) { }

        __device__
        fml operator()(const ssize_t& index)
        {
            fml denom = m_vect[(index / m_numInputRows)];

            if (denom > FML(0.0))
                return m_input[index] / denom;
            else
                return FML(1.0) / ((fml) m_numInputRows);
        }

    private:

        fml* m_input;
        fml* m_vect;
        u32  m_numInputRows;
};


tAnnLayerGPU::tAnnLayerGPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                           u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                           fml randWeightMin, fml randWeightMax)
    : tAnnLayerBase(type, rule, numInputDims, numNeurons, lcg,
                    randWeightMin, randWeightMax),
      m_cublasContext(NULL),
      m_gpu_w(NULL),
      m_gpu_b(NULL),
      m_gpu_dw_accum(NULL),
      m_gpu_db_accum(NULL),
      m_uniqueKeys(NULL),
      m_columnSums(NULL)
{
    s_createCublasContext(m_cublasContext);
}


tAnnLayerGPU::tAnnLayerGPU(iReadable* in)
    : tAnnLayerBase(in),
      m_cublasContext(NULL),
      m_gpu_w(NULL),
      m_gpu_b(NULL),
      m_gpu_dw_accum(NULL),
      m_gpu_db_accum(NULL),
      m_uniqueKeys(NULL),
      m_columnSums(NULL)
{
    s_createCublasContext(m_cublasContext);
}


tAnnLayerGPU::~tAnnLayerGPU()
{
    // The super d'tor are called automatically.

    s_cudaFree(m_gpu_w);
    s_cudaFree(m_gpu_b);
    s_cudaFree(m_gpu_dw_accum);
    s_cudaFree(m_gpu_db_accum);
    s_cudaFree(m_uniqueKeys);
    s_cudaFree(m_columnSums);

    s_destroyCublasContext(m_cublasContext);
}


void tAnnLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t*)m_cublasContext;

    if (!input)
        throw eInvalidArgument("The input matrix may not be null.");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims!");

    if (count == 0)
        throw eInvalidArgument("The count may not be zero.");

    if (!m_A || !m_a || count > m_maxCount)
    {
        m_maxCount = count;
        s_cudaFree(m_A);
        s_cudaFree(m_a);
        s_cudaFree(m_uniqueKeys);
        s_cudaFree(m_columnSums);
        m_A = s_cudaMalloc(m_numNeurons * m_maxCount);
        m_a = s_cudaMalloc(m_numNeurons * m_maxCount);
        m_uniqueKeys = s_cudaMalloc(m_maxCount);
        m_columnSums = s_cudaMalloc(m_maxCount);
        s_cudaFree(m_dA);
        s_cudaFree(m_prev_da);
    }
    m_curCount = count;

    if (!m_gpu_w || !m_gpu_b)
    {
        m_syncWeights_hostToDevice();
    }

    thrust::device_ptr<fml> A(m_A);
    thrust::device_ptr<fml> a(m_a);

    tFillColumnsWithFunc fillColumnsWith(m_gpu_b, m_numNeurons);
    thrust::tabulate(A, A+m_numNeurons*count, fillColumnsWith);

    fml n = FML(1.0) / ((fml) numInputDims);
    cublas_assert( cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m_numNeurons, count, numInputDims,
                               &n,
                               m_gpu_w, m_numNeurons,
                               input, numInputDims,
                               &n,
                               m_A, m_numNeurons), "cublasSgemm" );

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            tExpFunc expFunc;
            thrust::transform(A, A+m_numNeurons*count, a, expFunc);

            thrust::device_ptr<fml> uniqueKeys(m_uniqueKeys);
            thrust::device_ptr<fml> columnSums(m_columnSums);

            tColumnIndexFunc colIndexFunc(m_numNeurons);
            thrust::reduce_by_key(
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator((u32)0),
                    colIndexFunc),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator((u32)0),
                    colIndexFunc) + m_numNeurons*count,
                a,
                uniqueKeys,
                columnSums);

            tDivInputColsByVectorValues divInputColsByVectorValues(m_a, m_columnSums, m_numNeurons);
            thrust::tabulate(a, a + m_numNeurons*count, divInputColsByVectorValues);

            break;
        }

        case kLayerTypeLogistic:
        {
            tLogisticFunc func;
            thrust::transform(A, A+m_numNeurons*count, a, func);
            break;
        }

        case kLayerTypeHyperbolic:
        {
            tHyperbolicFunc func;
            thrust::transform(A, A+m_numNeurons*count, a, func);
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown layer type");
        }
    }
}


const fml* tAnnLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numNeurons;
    count = m_curCount;
    return m_a;
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
    u32 numWeights = m_numInputDims * m_numNeurons;
    if (!m_gpu_w)
        m_gpu_w = s_cudaMalloc(numWeights);
    cuda_assert( cudaMemcpy(m_gpu_w, m_w, numWeights*sizeof(fml), cudaMemcpyHostToDevice) );

    u32 numBiases = m_numNeurons;
    if (!m_gpu_b)
        m_gpu_b = s_cudaMalloc(numBiases);
    cuda_assert( cudaMemcpy(m_gpu_b, m_b, numBiases*sizeof(fml), cudaMemcpyHostToDevice) );
}


}   // namespace ml2
