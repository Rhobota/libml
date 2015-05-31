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


class tNoOp
{
    public:

        __device__
        fml operator()(const fml& val)
        {
            return val;
        }
};


template<class T>
class tMultWithUniOperator
{
    public:

        __host__ __device__
        fml operator()(const fml& da, const fml& A)
        {
            return da * m_uniOp(A);
        }

    private:

        T m_uniOp;
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


tAnnLayerGPU::tAnnLayerGPU(iReadable* in)
    : tAnnLayerBase(in),
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
    s_cudaFree(m_gpu_uniqueKeys);
    s_cudaFree(m_gpu_columnSums);
    s_cudaFree(m_gpu_ones_vector);

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

    if (!m_gpu_A || !m_gpu_a || count > m_maxCount)
    {
        m_maxCount = count;
        s_cudaFree(m_gpu_A);
        s_cudaFree(m_gpu_a);
        s_cudaFree(m_gpu_uniqueKeys);
        s_cudaFree(m_gpu_columnSums);
        m_gpu_A = s_cudaMalloc(m_numNeurons * m_maxCount);
        m_gpu_a = s_cudaMalloc(m_numNeurons * m_maxCount);
        m_gpu_uniqueKeys = s_cudaMalloc(m_maxCount);
        m_gpu_columnSums = s_cudaMalloc(m_maxCount);
        s_cudaFree(m_gpu_dA);
        s_cudaFree(m_gpu_prev_da);
        s_cudaFree(m_gpu_ones_vector);
    }
    m_curCount = count;

    if (!m_gpu_w || !m_gpu_b)
    {
        m_syncWeights_hostToDevice();
    }

    thrust::device_ptr<fml> A(m_gpu_A);
    thrust::device_ptr<fml> a(m_gpu_a);

    tFillColumnsWithFunc fillColumnsWith(m_gpu_b, m_numNeurons);
    thrust::tabulate(A, A+m_numNeurons*count, fillColumnsWith);

    fml n = FML(1.0) / ((fml) numInputDims);
    cublas_assert( cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m_numNeurons, count, numInputDims,
                               &n,
                               m_gpu_w, m_numNeurons,
                               input, numInputDims,
                               &n,
                               m_gpu_A, m_numNeurons), "cublasSgemm" );

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            tExpFunc expFunc;
            thrust::transform(A, A+m_numNeurons*count, a, expFunc);

            thrust::device_ptr<fml> uniqueKeys(m_gpu_uniqueKeys);
            thrust::device_ptr<fml> columnSums(m_gpu_columnSums);

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

            tDivInputColsByVectorValues divInputColsByVectorValues(m_gpu_a, m_gpu_columnSums, m_numNeurons);
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
    return m_gpu_a;
}


void tAnnLayerGPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t*)m_cublasContext;

    if (!outputErrorGradients)
        throw eInvalidArgument("outputErrorGradients may not be null!");

    if (numOutputDims != m_numNeurons)
        throw eInvalidArgument("Unexpected numOutputDims");

    if (outputCount != m_curCount)
        throw eInvalidArgument("Unexpected outputCount");

    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims");

    if (inputCount != m_curCount)
        throw eInvalidArgument("Unexpected inputCount");

    if (m_curCount == 0 || !m_gpu_A)
        throw eRuntimeError("What gives?");

    if (!m_gpu_dA)
    {
        m_gpu_dA = s_cudaMalloc(m_numNeurons * m_maxCount);
    }

    if (!m_gpu_prev_da)
    {
        m_gpu_prev_da = s_cudaMalloc(m_numInputDims * m_maxCount);
    }

    if (!m_gpu_ones_vector)
    {
        m_gpu_ones_vector = s_cudaMalloc(m_maxCount);
        thrust::device_ptr<fml> ones_vector(m_gpu_ones_vector);
        thrust::fill(ones_vector, ones_vector + m_maxCount, FML(1.0));
    }

    thrust::device_ptr<const fml>   da       (outputErrorGradients);   // numOutputDims x outputCount
    thrust::device_ptr<      fml>   dA       (m_gpu_dA);               // numOutputDims x outputCount
    thrust::device_ptr<      fml>   A        (m_gpu_A);                // numOutputDims x outputCount
    thrust::device_ptr<      fml>   prev_da  (m_gpu_prev_da);          // numInputDims x inputCount
    thrust::device_ptr<const fml>   inputMap (input);                  // numInputDims x inputCount
    thrust::device_ptr<      fml>   w        (m_gpu_w);                // numOutputDims x numInputDims
    thrust::device_ptr<      fml>   b        (m_gpu_b);                // numOutputDims x 1
    thrust::device_ptr<      fml>   dw_accum (m_gpu_dw_accum);         // numOutputDims x numInputDims
    thrust::device_ptr<      fml>   db_accum (m_gpu_db_accum);         // numOutputDims x 1

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            tNoOp func;
            thrust::transform(da, da + numOutputDims*outputCount, dA, func);
            break;
        }

        case kLayerTypeLogistic:
        {
            tMultWithUniOperator<tDirLogisticFunc> func;
            thrust::transform(da, da + numOutputDims*outputCount, A, dA, func);
            break;
        }

        case kLayerTypeHyperbolic:
        {
            tMultWithUniOperator<tDirHyperbolicFunc> func;
            thrust::transform(da, da + numOutputDims*outputCount, A, dA, func);
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown layer type");
            break;
        }
    }

    fml n = FML(1.0) / ((fml) numInputDims);
    fml zero = FML(0.0);

    if (calculateInputErrorGradients)
    {
        cublas_assert( cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   numInputDims, outputCount, numOutputDims,
                                   &n,
                                   m_gpu_w, numOutputDims,
                                   m_gpu_dA, numOutputDims,
                                   &zero,
                                   m_gpu_prev_da, numInputDims), "cublasSgemm" );
    }

    cublas_assert( cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                               numOutputDims, numInputDims, outputCount,
                               &n,
                               m_gpu_dA, numOutputDims,
                               input, numInputDims,
                               &zero,
                               m_gpu_dw_accum, numOutputDims), "cublasSgemm" );

    cublas_assert( cublasSgemv(*cublasHandle, CUBLAS_OP_N,
                               numOutputDims, outputCount,
                               &n,
                               m_gpu_dA, numOutputDims,
                               m_gpu_ones_vector, 1,
                               &zero,
                               m_gpu_db_accum, 1), "cublasSgemv" );

    fml batchSize = (fml) outputCount;

    switch (m_rule)
    {
        case kWeightUpRuleNone:
        {
            break;
        }

        case kWeightUpRuleFixedLearningRate:
        {
//          if (m_alpha <= FML(0.0))
//              throw eLogicError("When using the fixed learning rate rule, alpha must be set.");
//          fml mult = (FML(10.0) / batchSize) * m_alpha;
//          w -= mult * dw_accum;
//          b -= mult * db_accum;
            break;
        }

        case kWeightUpRuleMomentum:
        {
//          if (m_alpha <= FML(0.0))
//              throw eLogicError("When using the momentum update rule, alpha must be set.");
//          if (m_viscosity <= FML(0.0) || m_viscosity >= FML(1.0))
//              throw eLogicError("When using the momentum update rule, viscosity must be set.");
//          if (!m_gpu_vel)
//          {
//              u32 numWeights = (m_numInputDims+1) * m_numNeurons;  // <-- +1 to handle the b vector too
//              m_gpu_vel = new fml[numWeights];
//              for (u32 i = 0; i < numWeights; i++)
//                  m_gpu_vel[i] = FML(0.0);
//          }
//          fml mult = (FML(10.0) / batchSize) * m_alpha;
//          {
//              // Update w:
//              Map vel(m_gpu_vel, m_numNeurons, m_numInputDims);
//              vel *= m_viscosity;
//              vel -= mult*dw_accum;
//              w += vel;
//          }
//          {
//              // Update b:
//              Map vel(m_gpu_vel+m_numNeurons*m_numInputDims, m_numNeurons, 1);
//              vel *= m_viscosity;
//              vel -= mult*db_accum;
//              b += vel;
//          }
            break;
        }

        case kWeightUpRuleAdaptiveRates:
        {
            throw eNotImplemented("This used to be implemented in the old ANN... so look there as a reference if you want to implement it here again.");
        }

        case kWeightUpRuleRPROP:
        {
            throw eNotImplemented("This used to be implemented in the old ANN... so look there as a reference if you want to implement it here again.");
        }

        case kWeightUpRuleRMSPROP:
        {
//          if (m_alpha <= FML(0.0))
//              throw eLogicError("When using the rmsprop rule, alpha must be set.");
//          if (!m_gpu_dw_accum_avg)
//          {
//              u32 numWeights = (m_numInputDims+1) * m_numNeurons;  // <-- +1 to handle the b vector too
//              m_gpu_dw_accum_avg = new fml[numWeights];
//              for (u32 i = 0; i < numWeights; i++)
//                  m_gpu_dw_accum_avg[i] = FML(1000.0);
//          }
//          fml batchNormMult = FML(1.0) / batchSize;
//          {
//              // Update w:
//              Map dw_accum_avg(m_gpu_dw_accum_avg, m_numNeurons, m_numInputDims);
//              dw_accum *= batchNormMult;
//              dw_accum_avg *= FML(0.9);
//              dw_accum_avg += FML(0.1) * dw_accum.array().square().matrix();
//              w -= m_alpha * dw_accum.binaryExpr(dw_accum_avg, t_RMSPROP_update());
//          }
//          {
//              // Update b:
//              Map db_accum_avg(m_gpu_dw_accum_avg+m_numNeurons*m_numInputDims, m_numNeurons, 1);
//              db_accum *= batchNormMult;
//              db_accum_avg *= FML(0.9);
//              db_accum_avg += FML(0.1) * db_accum.array().square().matrix();
//              b -= m_alpha * db_accum.binaryExpr(db_accum_avg, t_RMSPROP_update());
//          }
            break;
        }

        case kWeightUpRuleARMS:
        {
            throw eNotImplemented("Not sure what I want here yet...");
        }

        default:
        {
            throw eRuntimeError("Unknown update rule");
        }
    }
}


const fml* tAnnLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_gpu_prev_da;
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


void tAnnLayerGPU::m_syncWeights_deviceToHost()  // TODO use this method!
{
    if (!m_gpu_w || !m_gpu_b)
        throw eRuntimeError("Cannot sync weight from device to host because there are no device weights!");

    u32 numWeights = m_numInputDims * m_numNeurons;
    cuda_assert( cudaMemcpy(m_w, m_gpu_w, numWeights*sizeof(fml), cudaMemcpyDeviceToHost) );

    u32 numBiases = m_numNeurons;
    cuda_assert( cudaMemcpy(m_b, m_gpu_b, numBiases*sizeof(fml), cudaMemcpyDeviceToHost) );
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
