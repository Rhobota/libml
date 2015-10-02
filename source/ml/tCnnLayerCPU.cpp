#include <ml/tCnnLayerCPU.h>
#include <ml/conv2d/cpu_optimized.h>

#include "Eigen.h"

#include "common_nn.ipp"


namespace ml
{


tCnnLayerCPU::tCnnLayerCPU()
    : tCnnLayerBase(),
      m_dw_accum(NULL),
      m_db_accum(NULL),
      m_A(NULL),
      m_a(NULL),
      m_dA(NULL),
      m_prev_da(NULL),
      m_vel(NULL),
      m_dw_accum_avg(NULL)
{
}


tCnnLayerCPU::tCnnLayerCPU(nLayerType type, nLayerWeightUpdateRule rule,
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
      m_dw_accum(NULL),
      m_db_accum(NULL),
      m_A(NULL),
      m_a(NULL),
      m_dA(NULL),
      m_prev_da(NULL),
      m_vel(NULL),
      m_dw_accum_avg(NULL)
{
    m_initAccum();
}


tCnnLayerCPU::~tCnnLayerCPU()
{
    // The super d'tor is called automatically.

    m_finalize();
}


void tCnnLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count,
                             bool isTrainMode, iLayer* prevLayer)
{
    if (!input)
        throw eInvalidArgument("The input matrix may not be null.");

    if (numInputDims != m_inputRows * m_inputCols * m_inputComponents)
        throw eInvalidArgument("Unexpected numInputDims!");

    if (count == 0)
        throw eInvalidArgument("The count may not be zero.");

    u32 numOutputDims = m_outputRows * m_outputCols * m_numKernels;

    if (!m_A || !m_a || count > m_maxCount)
    {
        m_maxCount = count;
        delete [] m_A;
        delete [] m_a;
        m_A = new fml[numOutputDims * m_maxCount];
        m_a = new fml[numOutputDims * m_maxCount];
        delete [] m_dA;
        delete [] m_prev_da;
        m_dA = NULL;
        m_prev_da = NULL;
    }
    m_curCount = count;

    Map A(m_A, numOutputDims, count);
    Map a(m_a, numOutputDims, count);

    fml n = FML(1.0) / ((fml) (m_kernelRows*m_kernelCols*m_inputComponents));

    conv2d::cpu_optimized::conv2d_multi_input(
            count, numInputDims, numOutputDims,
            input, m_inputRows, m_inputCols, m_inputComponents,
            m_w, m_kernelRows, m_kernelCols,
                 m_kernelStepY, m_kernelStepX,
                 m_numKernels,
            m_b, n,
            m_A);

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            throw eRuntimeError("A CNN softmax output layer makes no sense.");
            break;
        }

        case kLayerTypeLogistic:
        {
            tLogisticFunc func;
            a = A.unaryExpr(func);
            break;
        }

        case kLayerTypeHyperbolic:
        {
            tHyperbolicFunc func;
            a = A.unaryExpr(func);
            break;
        }

        case kLayerTypeReLU:
        {
            tReLUFunc func;
            a = A.unaryExpr(func);
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown layer type");
            break;
        }
    }
}


const fml* tCnnLayerCPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_outputRows * m_outputCols * m_numKernels;
    count = m_curCount;
    return m_a;
}


void tCnnLayerCPU::takeOutputErrorGradients(
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

    if (m_curCount == 0 || !m_A)
        throw eRuntimeError("What gives?");

    if (!m_dA)
        m_dA = new fml[numOutputDims * m_maxCount];

    if (!m_prev_da)
        m_prev_da = new fml[numInputDims * m_maxCount];

    MapConst da(outputErrorGradients, numOutputDims, outputCount);
    Map dA(m_dA, numOutputDims, outputCount);
    Map A(m_A, numOutputDims, outputCount);
    //Map prev_da(m_prev_da, numInputDims, inputCount);
    //MapConst inputMap(input, numInputDims, inputCount);

    Map w(m_w, m_kernelRows*m_kernelCols*m_inputComponents, m_numKernels);
    Map b(m_b, m_numKernels, 1);
    Map dw_accum(m_dw_accum, m_kernelRows*m_kernelCols*m_inputComponents, m_numKernels);
    Map db_accum(m_db_accum, m_numKernels, 1);

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            throw eRuntimeError("A CNN softmax output layer makes no sense.");
            break;
        }

        case kLayerTypeLogistic:
        {
            tDirLogisticFunc func;
            dA = (da.array() * A.unaryExpr(func).array()).matrix();
            break;
        }

        case kLayerTypeHyperbolic:
        {
            tDirHyperbolicFunc func;
            dA = (da.array() * A.unaryExpr(func).array()).matrix();
            break;
        }

        case kLayerTypeReLU:
        {
            tDirReLUFunc func;
            dA = (da.array() * A.unaryExpr(func).array()).matrix();
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown layer type");
            break;
        }
    }

    fml n = FML(1.0) / ((fml) (m_kernelRows*m_kernelCols*m_inputComponents));

    if (calculateInputErrorGradients)
    {
        conv2d::cpu_optimized::conv2d_backprop_multi_input(
                inputCount, numInputDims, numOutputDims,
                m_prev_da, m_inputRows, m_inputCols, m_inputComponents,
                m_w, m_kernelRows, m_kernelCols,
                     m_kernelStepY, m_kernelStepX,
                     m_numKernels,
                m_b, n,
                m_dA);
    }

    conv2d::cpu_optimized::conv2d_accumError_multi_input(
            inputCount, numInputDims, numOutputDims,
            input, m_inputRows, m_inputCols, m_inputComponents,
            m_dw_accum, m_kernelRows, m_kernelCols,
                        m_kernelStepY, m_kernelStepX,
                        m_numKernels,
            m_db_accum, n,
            m_dA);

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
            fml mult = (FML(1000.0) / batchSize) * m_alpha;
            w -= mult * dw_accum;
            b -= mult * db_accum;
            break;
        }

        case kWeightUpRuleMomentum:
        {
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the momentum update rule, alpha must be set.");
            if (m_viscosity <= FML(0.0) || m_viscosity >= FML(1.0))
                throw eLogicError("When using the momentum update rule, viscosity must be set.");
            if (!m_vel)
            {
                u32 numWeights = (m_kernelRows * m_kernelCols * m_inputComponents + 1) * m_numKernels;  // <-- +1 to handle the b vector too
                m_vel = new fml[numWeights];
                for (u32 i = 0; i < numWeights; i++)
                    m_vel[i] = FML(0.0);
            }
            fml mult = (FML(1000.0) / batchSize) * m_alpha;
            {
                // Update w:
                Map vel(m_vel, m_kernelRows*m_kernelCols*m_inputComponents, m_numKernels);
                vel *= m_viscosity;
                vel -= mult*dw_accum;
                w += vel;
            }
            {
                // Update b:
                Map vel(m_vel+m_kernelRows*m_kernelCols*m_inputComponents*m_numKernels, m_numKernels, 1);
                vel *= m_viscosity;
                vel -= mult*db_accum;
                b += vel;
            }
            break;
        }

        case kWeightUpRuleAdaptiveRates:
        {
            throw eNotImplemented("This used to be implemented in the old CNN... so look there as a reference if you want to implement it here again.");
            break;
        }

        case kWeightUpRuleRPROP:
        {
            throw eNotImplemented("This used to be implemented in the old CNN... so look there as a reference if you want to implement it here again.");
            break;
        }

        case kWeightUpRuleRMSPROP:
        {
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the rmsprop rule, alpha must be set.");
            if (!m_dw_accum_avg)
            {
                u32 numWeights = (m_kernelRows * m_kernelCols * m_inputComponents + 1) * m_numKernels;  // <-- +1 to handle the b vector too
                m_dw_accum_avg = new fml[numWeights];
                for (u32 i = 0; i < numWeights; i++)
                    m_dw_accum_avg[i] = FML(1000.0);
            }
            fml batchNormMult = FML(1.0) / batchSize;
            {
                // Update w:
                Map dw_accum_avg(m_dw_accum_avg, m_kernelRows*m_kernelCols*m_inputComponents, m_numKernels);
                dw_accum *= batchNormMult;
                dw_accum_avg *= FML(0.9);
                dw_accum_avg += FML(0.1) * dw_accum.array().square().matrix();
                w -= m_alpha * dw_accum.binaryExpr(dw_accum_avg, t_RMSPROP_update());
            }
            {
                // Update b:
                Map db_accum_avg(m_dw_accum_avg+m_kernelRows*m_kernelCols*m_inputComponents*m_numKernels, m_numKernels, 1);
                db_accum *= batchNormMult;
                db_accum_avg *= FML(0.9);
                db_accum_avg += FML(0.1) * db_accum.array().square().matrix();
                b -= m_alpha * db_accum.binaryExpr(db_accum_avg, t_RMSPROP_update());
            }
            break;
        }

        case kWeightUpRuleARMS:
        {
            throw eNotImplemented("Not sure what I want here yet...");
            break;
        }

        default:
        {
            throw eRuntimeError("Unknown update rule");
            break;
        }
    }
}


const fml* tCnnLayerCPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_inputRows * m_inputCols * m_inputComponents;
    count = m_curCount;
    return m_prev_da;
}


static
iLayer* s_newLayerFunc(iReadable* in)
{
    tCnnLayerCPU* layer = new tCnnLayerCPU();
    layer->unpack(in);
    return layer;
}


static u32 layerId = 465676;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tCnnLayerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


void tCnnLayerCPU::reset()
{
    m_finalize();
    tCnnLayerBase::reset();
    m_initAccum();
}


void tCnnLayerCPU::unpack(iReadable* in)
{
    tCnnLayerBase::unpack(in);
    m_finalize();
    m_initAccum();
}


void tCnnLayerCPU::m_initAccum()
{
    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;

    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    m_dw_accum = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_dw_accum[i] = FML(0.0);

    u32 numBiases = m_numKernels;
    m_db_accum = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_db_accum[i] = FML(0.0);
}


void tCnnLayerCPU::m_finalize()
{
    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;
    delete [] m_A; m_A = NULL;
    delete [] m_a; m_a = NULL;
    delete [] m_dA; m_dA = NULL;
    delete [] m_prev_da; m_prev_da = NULL;
    delete [] m_vel; m_vel = NULL;
    delete [] m_dw_accum_avg; m_dw_accum_avg = NULL;
}


}   // namespace ml
