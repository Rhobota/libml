#include <ml2/tAnnLayer.h>

#include "Eigen.h"


namespace ml2
{


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


class t_RMSPROP_wUpdate
{
    public:

        fml operator()(fml dw_accum, fml dw_accum_avg) const
        {
            return (dw_accum_avg > FML(0.0)) ? (dw_accum / std::sqrt(dw_accum_avg)) : FML(0.0);
        }
};


static
fml s_randInRange(algo::iLCG& lcg, fml randWeightMin, fml randWeightMax)
{
    u64 ra;
    fml rf;
    ra = lcg.next();
    rf = ((fml)ra) / ((fml)lcg.randMax());    // [0.0, 1.0]
    rf *= randWeightMax-randWeightMin;        // [0.0, rmax-rmin]
    rf += randWeightMin;                      // [rmin, rmax]
    return rf;
}


tAnnLayer::tAnnLayer(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                     u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                     fml randWeightMin, fml randWeightMax)
    : m_type(type),
      m_rule(rule),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_numInputDims(numInputDims),
      m_numNeurons(numNeurons),
      m_w(NULL),
      m_b(NULL),
      m_dw_accum(NULL),
      m_db_accum(NULL),
      m_curCount(0),
      m_maxCount(0),
      m_A(NULL),
      m_a(NULL),
      m_dA(NULL),
      m_prev_da(NULL),
      m_vel(NULL),
      m_dw_accum_avg(NULL)
{
    if (m_numInputDims == 0)
        throw eInvalidArgument("The number of input dimensions may not be zero.");
    if (m_numNeurons == 0)
        throw eInvalidArgument("The number of neurons may not be zero.");

    u32 numWeights = m_numInputDims * m_numNeurons;
    m_w = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_w[i] = s_randInRange(lcg, randWeightMin, randWeightMax);
    m_dw_accum = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_dw_accum[i] = FML(0.0);

    u32 numBiases = m_numNeurons;
    m_b = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_b[i] = s_randInRange(lcg, randWeightMin, randWeightMax);
    m_db_accum = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_db_accum[i] = FML(0.0);
}


tAnnLayer::~tAnnLayer()
{
    delete [] m_w; m_w = NULL;
    delete [] m_b; m_b = NULL;
    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;
    delete [] m_A; m_A = NULL;
    delete [] m_a; m_a = NULL;
    delete [] m_dA; m_dA = NULL;
    delete [] m_prev_da; m_prev_da = NULL;
    delete [] m_vel; m_vel = NULL;
    delete [] m_dw_accum_avg; m_dw_accum_avg = NULL;
}


void tAnnLayer::setAlpha(fml alpha)
{
    if (alpha <= FML(0.0))
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_alpha = alpha;
}


void tAnnLayer::setViscosity(fml viscosity)
{
    if (viscosity <= FML(0.0) || viscosity >= FML(1.0))
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_viscosity = viscosity;
}


void tAnnLayer::takeInput(const fml* input, u32 numInputDims, u32 count)
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

    A = (w * inputMap) / ((fml) numInputDims);
    for (u32 c = 0; c < count; c++)
        A.col(c) += b;

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


const fml* tAnnLayer::getOutput(u32& numOutputDims, u32& count) const
{
    numOutputDims = m_numNeurons;
    count = m_curCount;
    return m_a;
}


void tAnnLayer::takeOutputErrorGradients(
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
    Map w(m_w, outputCount, numInputDims);
    Map dw_accum(m_dw_accum, outputCount, numInputDims);
    Map db_accum(m_db_accum, outputCount, 1);

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

    if (calculateInputErrorGradients)
        prev_da = (w.transpose() * dA) / ((fml) numInputDims);

    dw_accum = (dA * inputMap.transpose()) / ((fml) numInputDims);
    db_accum = dA.rowwise().sum();

    fml batchSize = (fml) outputCount;

    // TODO -- update m_b

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
                u32 numWeights = m_numInputDims * m_numNeurons;
                m_vel = new fml[numWeights];
                for (u32 i = 0; i < numWeights; i++)
                    m_vel[i] = FML(0.0);
            }
            Map vel(m_vel, m_numNeurons, m_numInputDims);
            fml mult = (FML(10.0) / batchSize) * m_alpha;
            vel *= m_viscosity;
            vel -= mult*dw_accum;
            w += vel;
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
                u32 numWeights = m_numInputDims * m_numNeurons;
                m_dw_accum_avg = new fml[numWeights];
                for (u32 i = 0; i < numWeights; i++)
                    m_dw_accum_avg[i] = FML(1000.0);
            }
            Map dw_accum_avg(m_dw_accum_avg, m_numNeurons, m_numInputDims);
            fml batchNormMult = FML(1.0) / batchSize;
            dw_accum *= batchNormMult;
            dw_accum_avg *= FML(0.9);
            dw_accum_avg += FML(0.1) * dw_accum.array().square().matrix();
            w -= m_alpha * dw_accum.binaryExpr(dw_accum_avg, t_RMSPROP_wUpdate());
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


const fml* tAnnLayer::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    numInputDims = m_numInputDims;
    count = m_curCount;
    return m_prev_da;
}


void tAnnLayer::pack(iWritable* out) const
{
    // TODO
}


void tAnnLayer::unpack(iReadable* in)
{
    // TODO
}


}   // namespace ml2
