#include <ml2/tAnnLayerCPU.h>

#include <iomanip>

#include "Eigen.h"


namespace ml2
{


static
std::string s_layerTypeToString(tAnnLayerCPU::nAnnLayerType type)
{
    switch (type)
    {
        case tAnnLayerCPU::kLayerTypeLogistic:
            return "logistic";
        case tAnnLayerCPU::kLayerTypeHyperbolic:
            return "hyperbolic";
        case tAnnLayerCPU::kLayerTypeSoftmax:
            return "softmax";
        default:
            assert(false);
    }
}


static
char s_layerTypeToChar(tAnnLayerCPU::nAnnLayerType type)
{
    switch (type)
    {
        case tAnnLayerCPU::kLayerTypeLogistic:
            return 'l';
        case tAnnLayerCPU::kLayerTypeHyperbolic:
            return 'h';
        case tAnnLayerCPU::kLayerTypeSoftmax:
            return 's';
        default:
            assert(false);
    }
}


static
std::string s_weightUpRuleToString(tAnnLayerCPU::nAnnLayerWeightUpdateRule rule)
{
    switch (rule)
    {
        case tAnnLayerCPU::kWeightUpRuleNone:
            return "none";
        case tAnnLayerCPU::kWeightUpRuleFixedLearningRate:
            return "fixed rate";
        case tAnnLayerCPU::kWeightUpRuleMomentum:
            return "momentum";
        case tAnnLayerCPU::kWeightUpRuleAdaptiveRates:
            return "adaptive rates";
        case tAnnLayerCPU::kWeightUpRuleRPROP:
            return "rprop";
        case tAnnLayerCPU::kWeightUpRuleRMSPROP:
            return "rmsprop";
        case tAnnLayerCPU::kWeightUpRuleARMS:
            return "arms";
        default:
            assert(false);
    }
}


static
char s_weightUpRuleToChar(tAnnLayerCPU::nAnnLayerWeightUpdateRule rule)
{
    switch (rule)
    {
        case tAnnLayerCPU::kWeightUpRuleNone:
            return 'n';
        case tAnnLayerCPU::kWeightUpRuleFixedLearningRate:
            return 'f';
        case tAnnLayerCPU::kWeightUpRuleMomentum:
            return 'm';
        case tAnnLayerCPU::kWeightUpRuleAdaptiveRates:
            return 'a';
        case tAnnLayerCPU::kWeightUpRuleRPROP:
            return 'r';
        case tAnnLayerCPU::kWeightUpRuleRMSPROP:
            return 'R';
        case tAnnLayerCPU::kWeightUpRuleARMS:
            return 'A';
        default:
            assert(false);
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


tAnnLayerCPU::tAnnLayerCPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
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
      m_w_orig(NULL),
      m_b_orig(NULL),
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
    m_w_orig = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_w[i] = m_w_orig[i] = s_randInRange(lcg, randWeightMin, randWeightMax);
    m_dw_accum = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_dw_accum[i] = FML(0.0);

    u32 numBiases = m_numNeurons;
    m_b = new fml[numBiases];
    m_b_orig = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_b[i] = m_b_orig[i] = s_randInRange(lcg, randWeightMin, randWeightMax);
    m_db_accum = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_db_accum[i] = FML(0.0);
}


tAnnLayerCPU::tAnnLayerCPU(iReadable* in)
    : m_type(kLayerTypeLogistic),
      m_rule(kWeightUpRuleNone),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_numInputDims(0),
      m_numNeurons(0),
      m_w(NULL),
      m_b(NULL),
      m_w_orig(NULL),
      m_b_orig(NULL),
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
    this->unpack(in);
}


tAnnLayerCPU::~tAnnLayerCPU()
{
    delete [] m_w; m_w = NULL;
    delete [] m_b; m_b = NULL;
    delete [] m_w_orig; m_w_orig = NULL;
    delete [] m_b_orig; m_b_orig = NULL;
    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;
    delete [] m_A; m_A = NULL;
    delete [] m_a; m_a = NULL;
    delete [] m_dA; m_dA = NULL;
    delete [] m_prev_da; m_prev_da = NULL;
    delete [] m_vel; m_vel = NULL;
    delete [] m_dw_accum_avg; m_dw_accum_avg = NULL;
}


void tAnnLayerCPU::setAlpha(fml alpha)
{
    if (alpha <= FML(0.0))
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_alpha = alpha;
}


void tAnnLayerCPU::setViscosity(fml viscosity)
{
    if (viscosity <= FML(0.0) || viscosity >= FML(1.0))
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_viscosity = viscosity;
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


fml tAnnLayerCPU::calculateError(const tIO& output, const tIO& target)
{
    if (m_type == kLayerTypeSoftmax)
        return crossEntropyCost(output, target);
    else
        return standardSquaredError(output, target);
}


fml tAnnLayerCPU::calculateError(const std::vector<tIO>& outputs,
                                 const std::vector<tIO>& targets)
{
    if (m_type == kLayerTypeSoftmax)
        return crossEntropyCost(outputs, targets);
    else
        return standardSquaredError(outputs, targets);
}


void tAnnLayerCPU::reset()
{
    if (!m_w_orig || !m_b_orig)
        throw eRuntimeError("Cannot reset this ann layer because there is no original data. This is probably because you unpacked this layer from a stream.");

    u32 numWeights = m_numInputDims * m_numNeurons;
    for (u32 i = 0; i < numWeights; i++)
        m_w[i] = m_w_orig[i];

    u32 numBiases = m_numNeurons;
    for (u32 i = 0; i < numBiases; i++)
        m_b[i] = m_b_orig[i];
}


void tAnnLayerCPU::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    out << std::setw(w) << "ANN Layer:";
    out << std::setw(w) << s_layerTypeToString(m_type);
    out << std::setw(w) << s_weightUpRuleToString(m_rule);

    {
        std::ostringstream o;
        o << "a=" << m_alpha;
        if (m_rule == kWeightUpRuleMomentum)
            o << " v=" << m_viscosity;
        out << std::setw(w) << o.str();
    }

    out << std::setw(w) << m_numNeurons;

    out << std::endl;
}


std::string tAnnLayerCPU::layerInfoString() const
{
    std::ostringstream o;

    o << m_numNeurons;
    o << s_layerTypeToChar(m_type);
    o << '-';

    o << s_weightUpRuleToChar(m_rule);
    o << '-';

    o << "a" << m_alpha;
    if (m_rule == kWeightUpRuleMomentum)
        o << "v" << m_viscosity;

    return o.str();
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


static
void s_assert_writeAll(iWritable* out, const fml* buf, u32 size)
{
    for (u32 i = 0; i < size; i++)
        rho::pack(out, buf[i]);
}


static
void s_assert_readAll(iReadable* in, fml* buf, u32 size)
{
    for (u32 i = 0; i < size; i++)
        rho::unpack(in, buf[i]);
}


void tAnnLayerCPU::pack(iWritable* out) const
{
    rho::pack(out, (i32)m_type);
    rho::pack(out, (i32)m_rule);

    rho::pack(out, m_alpha);
    rho::pack(out, m_viscosity);

    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numNeurons);

    s_assert_writeAll(out, m_w, m_numInputDims * m_numNeurons);
    s_assert_writeAll(out, m_b, m_numNeurons);
}


void tAnnLayerCPU::unpack(iReadable* in)
{
    i32 type, rule;
    rho::unpack(in, type);
    rho::unpack(in, rule);
    m_type = (nAnnLayerType)type;
    m_rule = (nAnnLayerWeightUpdateRule)rule;

    rho::unpack(in, m_alpha);
    rho::unpack(in, m_viscosity);

    rho::unpack(in, m_numInputDims);
    if (m_numInputDims == 0)
        throw eInvalidArgument("The number of input dimensions may not be zero.");
    rho::unpack(in, m_numNeurons);
    if (m_numNeurons == 0)
        throw eInvalidArgument("The number of neurons may not be zero.");

    delete [] m_w; m_w = NULL;
    delete [] m_b; m_b = NULL;
    delete [] m_w_orig; m_w_orig = NULL;
    delete [] m_b_orig; m_b_orig = NULL;
    delete [] m_dw_accum; m_dw_accum = NULL;
    delete [] m_db_accum; m_db_accum = NULL;

    u32 numWeights = m_numInputDims * m_numNeurons;
    m_w = new fml[numWeights];
    s_assert_readAll(in, m_w, numWeights);
    m_dw_accum = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_dw_accum[i] = FML(0.0);

    u32 numBiases = m_numNeurons;
    m_b = new fml[numBiases];
    s_assert_readAll(in, m_b, numBiases);
    m_db_accum = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_db_accum[i] = FML(0.0);

    m_curCount = 0;
    m_maxCount = 0;

    delete [] m_A; m_A = NULL;
    delete [] m_a; m_a = NULL;
    delete [] m_dA; m_dA = NULL;
    delete [] m_prev_da; m_prev_da = NULL;
    delete [] m_vel; m_vel = NULL;
    delete [] m_dw_accum_avg; m_dw_accum_avg = NULL;
}


}   // namespace ml2
