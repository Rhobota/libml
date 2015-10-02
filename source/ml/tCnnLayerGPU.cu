#include <ml/tCnnLayerGPU.h>
#include <ml/conv2d/gpu.h>

#define ENABLE_DEVICE_FUNCTIONS
#include "common_nn.ipp"

#include <cassert>
#include <iostream>


namespace ml
{


tCnnLayerGPU::tCnnLayerGPU()
    : tCnnLayerBase(),
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
    m_syncWeights_hostToDevice();
    m_initAccum();
}


tCnnLayerGPU::~tCnnLayerGPU()
{
    // The super d'tor is called automatically.

    m_finalize();
}


void tCnnLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count,
                             bool isTrainMode, iLayer* prevLayer)
{
    if (!input)
        throw eInvalidArgument("The input matrix may not be null.");

    if (numInputDims != m_inputRows * m_inputCols * m_inputComponents)
        throw eInvalidArgument("Unexpected numInputDims!");

    if (count == 0)
        throw eInvalidArgument("The count may not be zero.");

    u32 numOutputDims = m_outputRows * m_outputCols * m_numKernels;

    if (!m_gpu_A || !m_gpu_a || count > m_maxCount)
    {
        m_maxCount = count;
        s_cudaFree(m_gpu_A);
        s_cudaFree(m_gpu_a);
        m_gpu_A = s_cudaMalloc(numOutputDims * m_maxCount);
        m_gpu_a = s_cudaMalloc(numOutputDims * m_maxCount);
        s_cudaFree(m_gpu_dA);
        s_cudaFree(m_gpu_prev_da);
        m_gpu_dA = NULL;
        m_gpu_prev_da = NULL;
    }
    m_curCount = count;

    thrust::device_ptr<fml> A(m_gpu_A);    // numOutputDims x count
    thrust::device_ptr<fml> a(m_gpu_a);    // numOutputDims x count

    fml n = FML(1.0) / ((fml) (m_kernelRows*m_kernelCols*m_inputComponents));

    conv2d::gpu::conv2d_multi_input(
            count, numInputDims, numOutputDims,
            input, m_inputRows, m_inputCols, m_inputComponents,
            m_gpu_w, m_kernelRows, m_kernelCols,
                     m_kernelStepY, m_kernelStepX,
                     m_numKernels,
            m_gpu_b, n,
            m_gpu_A);

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            throw eRuntimeError("A CNN softmax output layer makes no sense.");
        }

        case kLayerTypeLogistic:
        {
            tLogisticFunc func;
            thrust::transform(A, A+numOutputDims*count, a, func);
            break;
        }

        case kLayerTypeHyperbolic:
        {
            tHyperbolicFunc func;
            thrust::transform(A, A+numOutputDims*count, a, func);
            break;
        }

        case kLayerTypeReLU:
        {
            tReLUFunc func;
            thrust::transform(A, A+numOutputDims*count, a, func);
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown layer type");
        }
    }
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
    if (!outputErrorGradients)
        throw eInvalidArgument("outputErrorGradients may not be null!");

    if (numOutputDims != m_outputRows * m_outputCols * m_numKernels)
        throw eInvalidArgument("Unexpected numOutputDims");

    if (outputCount != m_curCount)
        throw eInvalidArgument("Unexpected outputCount");

    if (!input)
        throw eInvalidArgument("input may not be null!");

    if (numInputDims != m_inputRows * m_inputCols * m_inputComponents)
        throw eInvalidArgument("Unexpected numInputDims");

    if (inputCount != m_curCount)
        throw eInvalidArgument("Unexpected inputCount");

    if (m_curCount == 0 || !m_gpu_A)
        throw eRuntimeError("What gives?");

    if (!m_gpu_dA)
        m_gpu_dA = s_cudaMalloc(numOutputDims * m_maxCount);

    if (!m_gpu_prev_da)
        m_gpu_prev_da = s_cudaMalloc(numInputDims * m_maxCount);

    thrust::device_ptr<const fml>   da       (outputErrorGradients);   // numOutputDims x outputCount
    thrust::device_ptr<      fml>   dA       (m_gpu_dA);               // numOutputDims x outputCount
    thrust::device_ptr<      fml>   A        (m_gpu_A);                // numOutputDims x outputCount
    //thrust::device_ptr<      fml>   prev_da  (m_gpu_prev_da);          // numInputDims x inputCount
    //thrust::device_ptr<const fml>   inputMap (input);                  // numInputDims x inputCount

    thrust::device_ptr<      fml>   w        (m_gpu_w);                // m_kernelRows*m_kernelCols*m_inputComponents x m_numKernels
    thrust::device_ptr<      fml>   b        (m_gpu_b);                // m_numKernels x 1
    thrust::device_ptr<      fml>   dw_accum (m_gpu_dw_accum);         // m_kernelRows*m_kernelCols*m_inputComponents x m_numKernels
    thrust::device_ptr<      fml>   db_accum (m_gpu_db_accum);         // m_numKernels x 1

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            throw eRuntimeError("A CNN softmax output layer makes no sense.");
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

    fml n = FML(1.0) / ((fml) (m_kernelRows*m_kernelCols*m_inputComponents));

    if (calculateInputErrorGradients)
    {
        conv2d::gpu::conv2d_backprop_multi_input(
                inputCount, numInputDims, numOutputDims,
                m_gpu_prev_da, m_inputRows, m_inputCols, m_inputComponents,
                m_gpu_w, m_kernelRows, m_kernelCols,
                         m_kernelStepY, m_kernelStepX,
                         m_numKernels,
                m_gpu_b, n,
                m_gpu_dA);
    }

    conv2d::gpu::conv2d_accumError_multi_input(
            inputCount, numInputDims, numOutputDims,
            input, m_inputRows, m_inputCols, m_inputComponents,
            m_gpu_dw_accum, m_kernelRows, m_kernelCols,
                            m_kernelStepY, m_kernelStepX,
                            m_numKernels,
            m_gpu_db_accum, n,
            m_gpu_dA);

    fml batchSize = (fml) outputCount;
    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    u32 numBiases = m_numKernels;

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
            fml mult = (FML(1000.0) / batchSize) * m_alpha;
            tSubWithScalarMult func(mult);
            thrust::transform(w, w + numWeights, dw_accum, w, func);
            thrust::transform(b, b + numBiases, db_accum, b, func);
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
                m_gpu_vel = s_cudaMalloc(numWeights + numBiases);
                thrust::device_ptr<fml> vel(m_gpu_vel);
                thrust::fill(vel, vel + numWeights + numBiases, FML(0.0));
            }
            fml mult = (FML(1000.0) / batchSize) * m_alpha;
            tVelUpdateFunc velUpdateFunc(m_viscosity, mult);
            {
                // Update w:
                thrust::device_ptr<fml> vel(m_gpu_vel);
                thrust::transform(vel, vel + numWeights, dw_accum, vel, velUpdateFunc);
                thrust::transform(w, w + numWeights, vel, w, thrust::plus<fml>());
            }
            {
                // Update b:
                thrust::device_ptr<fml> vel(m_gpu_vel + numWeights);
                thrust::transform(vel, vel + numBiases, db_accum, vel, velUpdateFunc);
                thrust::transform(b, b + numBiases, vel, b, thrust::plus<fml>());
            }
            break;
        }

        case kWeightUpRuleAdaptiveRates:
        {
            throw eNotImplemented("This used to be implemented in the old CNN... so look there as a reference if you want to implement it here again.");
        }

        case kWeightUpRuleRPROP:
        {
            throw eNotImplemented("This used to be implemented in the old CNN... so look there as a reference if you want to implement it here again.");
        }

        case kWeightUpRuleRMSPROP:
        {
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the rmsprop rule, alpha must be set.");
            if (!m_gpu_dw_accum_avg)
            {
                m_gpu_dw_accum_avg = s_cudaMalloc(numWeights + numBiases);
                thrust::device_ptr<fml> dw_accum_avg(m_gpu_dw_accum_avg);
                thrust::fill(dw_accum_avg, dw_accum_avg + numWeights + numBiases, FML(1000.0));
            }
            fml batchNormMult = FML(1.0) / batchSize;
            tMultBy batchNormFunc(batchNormMult);
            t_RMSPROP_avg_update avgUpdateFunc;
            t_RMSPROP_update_with_alpha rmsPropUpdateFunc(m_alpha);
            {
                // Update w:
                thrust::device_ptr<fml> dw_accum_avg(m_gpu_dw_accum_avg);
                thrust::transform(dw_accum, dw_accum + numWeights, dw_accum, batchNormFunc);
                thrust::transform(dw_accum_avg, dw_accum_avg + numWeights, dw_accum, dw_accum_avg, avgUpdateFunc);
                thrust::transform(dw_accum, dw_accum + numWeights, dw_accum_avg, dw_accum, rmsPropUpdateFunc);
                thrust::transform(w, w + numWeights, dw_accum, w, thrust::minus<fml>());
            }
            {
                // Update b:
                thrust::device_ptr<fml> db_accum_avg(m_gpu_dw_accum_avg + numWeights);
                thrust::transform(db_accum, db_accum + numBiases, db_accum, batchNormFunc);
                thrust::transform(db_accum_avg, db_accum_avg + numBiases, db_accum, db_accum_avg, avgUpdateFunc);
                thrust::transform(db_accum, db_accum + numBiases, db_accum_avg, db_accum, rmsPropUpdateFunc);
                thrust::transform(b, b + numBiases, db_accum, b, thrust::minus<fml>());
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


}   // namespace ml
