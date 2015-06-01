#include <ml2/tAnnLayerCPU.h>

#include "../ml/Eigen.h"


namespace ml2
{


#include "common.ipp"


tAnnLayerCPU::tAnnLayerCPU()
    : tAnnLayerBase(),
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


tAnnLayerCPU::tAnnLayerCPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                           u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                           fml randWeightMin, fml randWeightMax)
    : tAnnLayerBase(type, rule, numInputDims, numNeurons, lcg,
                    randWeightMin, randWeightMax),
      m_dw_accum(NULL),
      m_db_accum(NULL),
      m_A(NULL),
      m_a(NULL),
      m_dA(NULL),
      m_prev_da(NULL),
      m_vel(NULL),
      m_dw_accum_avg(NULL)
{
    u32 numWeights = m_numInputDims * m_numNeurons;
    m_dw_accum = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_dw_accum[i] = FML(0.0);

    u32 numBiases = m_numNeurons;
    m_db_accum = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_db_accum[i] = FML(0.0);
}


tAnnLayerCPU::~tAnnLayerCPU()
{
    // The super d'tor is called automatically.

    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;
    delete [] m_A; m_A = NULL;
    delete [] m_a; m_a = NULL;
    delete [] m_dA; m_dA = NULL;
    delete [] m_prev_da; m_prev_da = NULL;
    delete [] m_vel; m_vel = NULL;
    delete [] m_dw_accum_avg; m_dw_accum_avg = NULL;
}


void tAnnLayerCPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    if (!input)
        throw eInvalidArgument("The input matrix may not be null.");

    if (numInputDims != m_numInputDims)
        throw eInvalidArgument("Unexpected numInputDims!");

    if (count == 0)
        throw eInvalidArgument("The count may not be zero.");

    if (!m_A || !m_a || count > m_maxCount)
    {
        m_maxCount = count;
        delete [] m_A;
        delete [] m_a;
        m_A = new fml[m_numNeurons * m_maxCount];
        m_a = new fml[m_numNeurons * m_maxCount];
        delete [] m_dA;
        delete [] m_prev_da;
        m_dA = NULL;
        m_prev_da = NULL;
    }
    m_curCount = count;

    MapConst inputMap(input, numInputDims, count);
    Map w(m_w, m_numNeurons, numInputDims);
    Map b(m_b, m_numNeurons, 1);
    Map A(m_A, m_numNeurons, count);
    Map a(m_a, m_numNeurons, count);

    fml n = FML(1.0) / ((fml) numInputDims);

    A.noalias() = n * w * inputMap;
    for (u32 c = 0; c < count; c++)
        A.col(c) += n * b;

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            tExpFunc func;
            a = A.unaryExpr(func);
            for (i32 c = 0; c < a.cols(); c++)
            {
                fml denom = a.col(c).sum();
                if (denom > FML(0.0))
                    a.col(c) /= denom;
                else
                    a.col(c).setConstant(FML(1.0) / (fml)a.rows());
            }
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

        default:
        {
            throw eRuntimeError("Unknown layer type");
            break;
        }
    }
}


const fml* tAnnLayerCPU::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numNeurons;
    count = m_curCount;
    return m_a;
}


void tAnnLayerCPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
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

    if (m_curCount == 0 || !m_A)
        throw eRuntimeError("What gives?");

    if (!m_dA)
        m_dA = new fml[m_numNeurons * m_maxCount];

    if (!m_prev_da)
        m_prev_da = new fml[m_numInputDims * m_maxCount];

    MapConst da(outputErrorGradients, numOutputDims, outputCount);
    Map dA(m_dA, numOutputDims, outputCount);
    Map A(m_A, numOutputDims, outputCount);
    Map prev_da(m_prev_da, numInputDims, inputCount);
    MapConst inputMap(input, numInputDims, inputCount);
    Map w(m_w, numOutputDims, numInputDims);
    Map b(m_b, numOutputDims, 1);
    Map dw_accum(m_dw_accum, numOutputDims, numInputDims);
    Map db_accum(m_db_accum, numOutputDims, 1);

    switch (m_type)
    {
        case kLayerTypeSoftmax:
        {
            dA = da;
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

        default:
        {
            throw eRuntimeError("Unknown layer type");
            break;
        }
    }

    fml n = FML(1.0) / ((fml) numInputDims);

    if (calculateInputErrorGradients)
        prev_da.noalias() = n * w.transpose() * dA;

    dw_accum.noalias() = n * dA * inputMap.transpose();
    db_accum = n * dA.rowwise().sum();

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
                u32 numWeights = (m_numInputDims+1) * m_numNeurons;  // <-- +1 to handle the b vector too
                m_vel = new fml[numWeights];
                for (u32 i = 0; i < numWeights; i++)
                    m_vel[i] = FML(0.0);
            }
            fml mult = (FML(10.0) / batchSize) * m_alpha;
            {
                // Update w:
                Map vel(m_vel, m_numNeurons, m_numInputDims);
                vel *= m_viscosity;
                vel -= mult*dw_accum;
                w += vel;
            }
            {
                // Update b:
                Map vel(m_vel+m_numNeurons*m_numInputDims, m_numNeurons, 1);
                vel *= m_viscosity;
                vel -= mult*db_accum;
                b += vel;
            }
            break;
        }

        case kWeightUpRuleAdaptiveRates:
        {
            throw eNotImplemented("This used to be implemented in the old ANN... so look there as a reference if you want to implement it here again.");
            break;
        }

        case kWeightUpRuleRPROP:
        {
            throw eNotImplemented("This used to be implemented in the old ANN... so look there as a reference if you want to implement it here again.");
            break;
        }

        case kWeightUpRuleRMSPROP:
        {
            if (m_alpha <= FML(0.0))
                throw eLogicError("When using the rmsprop rule, alpha must be set.");
            if (!m_dw_accum_avg)
            {
                u32 numWeights = (m_numInputDims+1) * m_numNeurons;  // <-- +1 to handle the b vector too
                m_dw_accum_avg = new fml[numWeights];
                for (u32 i = 0; i < numWeights; i++)
                    m_dw_accum_avg[i] = FML(1000.0);
            }
            fml batchNormMult = FML(1.0) / batchSize;
            {
                // Update w:
                Map dw_accum_avg(m_dw_accum_avg, m_numNeurons, m_numInputDims);
                dw_accum *= batchNormMult;
                dw_accum_avg *= FML(0.9);
                dw_accum_avg += FML(0.1) * dw_accum.array().square().matrix();
                w -= m_alpha * dw_accum.binaryExpr(dw_accum_avg, t_RMSPROP_update());
            }
            {
                // Update b:
                Map db_accum_avg(m_dw_accum_avg+m_numNeurons*m_numInputDims, m_numNeurons, 1);
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


const fml* tAnnLayerCPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_prev_da;
}


static
iLayer* s_newLayerFunc(iReadable* in)
{
    return new tAnnLayerCPU(in);
}


static u32 layerId = 27424;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tAnnLayerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


void tAnnLayerCPU::unpack(iReadable* in)
{
    tAnnLayerBase::unpack(in);

    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;

    u32 numWeights = m_numInputDims * m_numNeurons;
    m_dw_accum = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_dw_accum[i] = FML(0.0);

    u32 numBiases = m_numNeurons;
    m_db_accum = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_db_accum[i] = FML(0.0);

    delete [] m_A; m_A = NULL;
    delete [] m_a; m_a = NULL;
    delete [] m_dA; m_dA = NULL;
    delete [] m_prev_da; m_prev_da = NULL;
    delete [] m_vel; m_vel = NULL;
    delete [] m_dw_accum_avg; m_dw_accum_avg = NULL;
}


}   // namespace ml2
