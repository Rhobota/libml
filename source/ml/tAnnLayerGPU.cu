#include <ml/tAnnLayerGPU.h>

#define ENABLE_DEVICE_FUNCTIONS
#include "common_nn.ipp"

#include <cassert>
#include <iostream>


namespace ml
{


tAnnLayerGPU::tAnnLayerGPU()
    : tAnnLayerBase(),
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


tAnnLayerGPU::tAnnLayerGPU(nLayerType type, nLayerWeightUpdateRule rule,
                           u32 inputRows, u32 inputCols, u32 inputComponents,
                           u32 numNeurons,
                           algo::iLCG& lcg, fml randWeightMin, fml randWeightMax)
    : tAnnLayerBase(type, rule, inputRows, inputCols, inputComponents,
                    numNeurons, lcg, randWeightMin, randWeightMax),
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


tAnnLayerGPU::~tAnnLayerGPU()
{
    // The super d'tor are called automatically.

    m_finalize();
}


void tAnnLayerGPU::currentState(std::vector<tIO>& weights, tIO& biases, tIO& outputs) const
{
    if (m_curCount == 0)
        throw eRuntimeError("This layer has never seen an input. This method cannot populate 'outputs'.");

    u32 numNeurons = m_numNeurons;
    u32 numInputDims = m_numInputDims;

    weights.resize(numNeurons);
    biases.resize(numNeurons);
    outputs.resize(numNeurons);

    m_syncWeights_deviceToHost();
    u32 outputStart = (m_curCount - 1) * numNeurons;
    s_cudaCopyDeviceToHost(&outputs[0], m_gpu_a + outputStart, numNeurons);

    for (u32 nn = 0; nn < numNeurons; nn++)
    {
        tIO& weightsHere = weights[nn];
        weightsHere.resize(numInputDims);

        for (u32 ii = 0; ii < numInputDims; ii++)
            weightsHere[ii] = m_w[ii*numNeurons + nn];

        biases[nn] = m_b[nn];
    }
}


void tAnnLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count,
                             bool isTrainMode, iLayer* prevLayer)
{
    cublasHandle_t* cublasHandle = (cublasHandle_t*)m_cublasContext;

    if (!input)
        throw eInvalidArgument("The input matrix may not be null.");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims!");

    if (count == 0)
        throw eInvalidArgument("The count may not be zero.");

    if (!m_gpu_w || !m_gpu_b)
        throw eRuntimeError("How are the gpu weight now initialized yet?");

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

        case kLayerTypeReLU:
        {
            tReLUFunc func;
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

    if (!m_gpu_w || !m_gpu_b)
        throw eRuntimeError("How are the gpu weight now initialized yet?");

    if (!m_gpu_dA)
        m_gpu_dA = s_cudaMalloc(m_numNeurons * m_maxCount);

    if (!m_gpu_prev_da)
        m_gpu_prev_da = s_cudaMalloc(m_numInputDims * m_maxCount);

    if (!m_gpu_ones_vector)
    {
        m_gpu_ones_vector = s_cudaMalloc(m_maxCount);
        thrust::device_ptr<fml> ones_vector(m_gpu_ones_vector);
        thrust::fill(ones_vector, ones_vector + m_maxCount, FML(1.0));
    }

    thrust::device_ptr<const fml>   da       (outputErrorGradients);   // numOutputDims x outputCount
    thrust::device_ptr<      fml>   dA       (m_gpu_dA);               // numOutputDims x outputCount
    thrust::device_ptr<      fml>   A        (m_gpu_A);                // numOutputDims x outputCount
    //thrust::device_ptr<      fml>   prev_da  (m_gpu_prev_da);          // numInputDims x inputCount
    //thrust::device_ptr<const fml>   inputMap (input);                  // numInputDims x inputCount
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
            tMultWithDirLogisticFunc func;
            thrust::transform(da, da + numOutputDims*outputCount, A, dA, func);
            break;
        }

        case kLayerTypeHyperbolic:
        {
            tMultWithDirHyperbolicFunc func;
            thrust::transform(da, da + numOutputDims*outputCount, A, dA, func);
            break;
        }

        case kLayerTypeReLU:
        {
            tMultWithDirReLUFunc func;
            thrust::transform(da, da + numOutputDims*outputCount, A, dA, func);
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown layer type");
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
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the fixed learning rate rule, alpha must be set.");
            fml mult = (FML(10.0) / batchSize) * m_alpha;
            tSubWithScalarMult func(mult);
            thrust::transform(w, w + numOutputDims*numInputDims, dw_accum, w, func);
            thrust::transform(b, b + numOutputDims, db_accum, b, func);
            break;
        }

        case kWeightUpRuleMomentum:
        {
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the momentum update rule, alpha must be set.");
            if (m_viscosity <= FML(0.0) || m_viscosity >= FML(1.0))
                throw eLogicError("When using the momentum update rule, viscosity must be set.");
            if (!m_gpu_vel)
            {
                u32 numWeights = (m_numInputDims+1) * m_numNeurons;  // <-- +1 to handle the b vector too
                m_gpu_vel = s_cudaMalloc(numWeights);
                thrust::device_ptr<fml> vel(m_gpu_vel);
                thrust::fill(vel, vel + numWeights, FML(0.0));
            }
            fml mult = (FML(10.0) / batchSize) * m_alpha;
            tVelUpdateFunc velUpdateFunc(m_viscosity, mult);
            {
                // Update w:
                thrust::device_ptr<fml> vel(m_gpu_vel);
                thrust::transform(vel, vel + numOutputDims*numInputDims, dw_accum, vel, velUpdateFunc);
                thrust::transform(w, w + numOutputDims*numInputDims, vel, w, thrust::plus<fml>());
            }
            {
                // Update b:
                thrust::device_ptr<fml> vel(m_gpu_vel + numOutputDims*numInputDims);
                thrust::transform(vel, vel + numOutputDims, db_accum, vel, velUpdateFunc);
                thrust::transform(b, b + numOutputDims, vel, b, thrust::plus<fml>());
            }
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
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the rmsprop rule, alpha must be set.");
            if (!m_gpu_dw_accum_avg)
            {
                u32 numWeights = (m_numInputDims+1) * m_numNeurons;  // <-- +1 to handle the b vector too
                m_gpu_dw_accum_avg = s_cudaMalloc(numWeights);
                thrust::device_ptr<fml> dw_accum_avg(m_gpu_dw_accum_avg);
                thrust::fill(dw_accum_avg, dw_accum_avg + numWeights, FML(1000.0));
            }
            fml batchNormMult = FML(1.0) / batchSize;
            tMultBy batchNormFunc(batchNormMult);
            t_RMSPROP_avg_update avgUpdateFunc;
            t_RMSPROP_update_with_alpha rmsPropUpdateFunc(m_alpha);
            {
                // Update w:
                thrust::device_ptr<fml> dw_accum_avg(m_gpu_dw_accum_avg);
                thrust::transform(dw_accum, dw_accum + numOutputDims*numInputDims, dw_accum, batchNormFunc);
                thrust::transform(dw_accum_avg, dw_accum_avg + numOutputDims*numInputDims, dw_accum, dw_accum_avg, avgUpdateFunc);
                thrust::transform(dw_accum, dw_accum + numOutputDims*numInputDims, dw_accum_avg, dw_accum, rmsPropUpdateFunc);
                thrust::transform(w, w + numOutputDims*numInputDims, dw_accum, w, thrust::minus<fml>());
            }
            {
                // Update b:
                thrust::device_ptr<fml> db_accum_avg(m_gpu_dw_accum_avg + numOutputDims*numInputDims);
                thrust::transform(db_accum, db_accum + numOutputDims, db_accum, batchNormFunc);
                thrust::transform(db_accum_avg, db_accum_avg + numOutputDims, db_accum, db_accum_avg, avgUpdateFunc);
                thrust::transform(db_accum, db_accum + numOutputDims, db_accum_avg, db_accum, rmsPropUpdateFunc);
                thrust::transform(b, b + numOutputDims, db_accum, b, thrust::minus<fml>());
            }
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
    tAnnLayerGPU* layer = new tAnnLayerGPU();
    layer->unpack(in);
    return layer;
}


static u32 layerId = 78879;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tAnnLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


void tAnnLayerGPU::reset()
{
    m_finalize();
    tAnnLayerBase::reset();
    s_createCublasContext(m_cublasContext);
    m_syncWeights_hostToDevice();
    m_initAccum();
}


void tAnnLayerGPU::pack(iWritable* out) const
{
    m_syncWeights_deviceToHost();
    tAnnLayerBase::pack(out);
}


void tAnnLayerGPU::unpack(iReadable* in)
{
    tAnnLayerBase::unpack(in);
    m_finalize();
    s_createCublasContext(m_cublasContext);
    m_syncWeights_hostToDevice();
    m_initAccum();
}


void tAnnLayerGPU::m_initAccum()
{
    s_cudaFree(m_gpu_dw_accum);
    s_cudaFree(m_gpu_db_accum);

    u32 numWeights = m_numInputDims * m_numNeurons;
    m_gpu_dw_accum = s_cudaMalloc(numWeights);
    thrust::device_ptr<fml> dw_accum(m_gpu_dw_accum);
    thrust::fill(dw_accum, dw_accum + numWeights, FML(0.0));

    u32 numBiases = m_numNeurons;
    m_gpu_db_accum = s_cudaMalloc(numBiases);
    thrust::device_ptr<fml> db_accum(m_gpu_db_accum);
    thrust::fill(db_accum, db_accum + numBiases, FML(0.0));
}


void tAnnLayerGPU::m_finalize()
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


void tAnnLayerGPU::m_syncWeights_deviceToHost() const
{
    if (!m_gpu_w || !m_gpu_b)
        throw eRuntimeError("Cannot sync weight from device to host because there are no device weights!");

    u32 numWeights = m_numInputDims * m_numNeurons;
    s_cudaCopyDeviceToHost(m_w, m_gpu_w, numWeights);

    u32 numBiases = m_numNeurons;
    s_cudaCopyDeviceToHost(m_b, m_gpu_b, numBiases);
}


void tAnnLayerGPU::m_syncWeights_hostToDevice()
{
    u32 numWeights = m_numInputDims * m_numNeurons;
    s_cudaFree(m_gpu_w);
    m_gpu_w = s_cudaMalloc(numWeights);
    s_cudaCopyHostToDevice(m_gpu_w, m_w, numWeights);

    u32 numBiases = m_numNeurons;
    s_cudaFree(m_gpu_b);
    m_gpu_b = s_cudaMalloc(numBiases);
    s_cudaCopyHostToDevice(m_gpu_b, m_b, numBiases);
}


}   // namespace ml
