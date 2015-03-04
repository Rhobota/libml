#include <ml2/tAnnLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml2
{


static
std::string s_layerTypeToString(tAnnLayerBase::nAnnLayerType type)
{
    switch (type)
    {
        case tAnnLayerBase::kLayerTypeLogistic:
            return "logistic";
        case tAnnLayerBase::kLayerTypeHyperbolic:
            return "hyperbolic";
        case tAnnLayerBase::kLayerTypeSoftmax:
            return "softmax";
        default:
            assert(false);
    }
}


static
char s_layerTypeToChar(tAnnLayerBase::nAnnLayerType type)
{
    switch (type)
    {
        case tAnnLayerBase::kLayerTypeLogistic:
            return 'l';
        case tAnnLayerBase::kLayerTypeHyperbolic:
            return 'h';
        case tAnnLayerBase::kLayerTypeSoftmax:
            return 's';
        default:
            assert(false);
    }
}


static
std::string s_weightUpRuleToString(tAnnLayerBase::nAnnLayerWeightUpdateRule rule)
{
    switch (rule)
    {
        case tAnnLayerBase::kWeightUpRuleNone:
            return "none";
        case tAnnLayerBase::kWeightUpRuleFixedLearningRate:
            return "fixed rate";
        case tAnnLayerBase::kWeightUpRuleMomentum:
            return "momentum";
        case tAnnLayerBase::kWeightUpRuleAdaptiveRates:
            return "adaptive rates";
        case tAnnLayerBase::kWeightUpRuleRPROP:
            return "rprop";
        case tAnnLayerBase::kWeightUpRuleRMSPROP:
            return "rmsprop";
        case tAnnLayerBase::kWeightUpRuleARMS:
            return "arms";
        default:
            assert(false);
    }
}


static
char s_weightUpRuleToChar(tAnnLayerBase::nAnnLayerWeightUpdateRule rule)
{
    switch (rule)
    {
        case tAnnLayerBase::kWeightUpRuleNone:
            return 'n';
        case tAnnLayerBase::kWeightUpRuleFixedLearningRate:
            return 'f';
        case tAnnLayerBase::kWeightUpRuleMomentum:
            return 'm';
        case tAnnLayerBase::kWeightUpRuleAdaptiveRates:
            return 'a';
        case tAnnLayerBase::kWeightUpRuleRPROP:
            return 'r';
        case tAnnLayerBase::kWeightUpRuleRMSPROP:
            return 'R';
        case tAnnLayerBase::kWeightUpRuleARMS:
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


tAnnLayerBase::tAnnLayerBase(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
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


tAnnLayerBase::tAnnLayerBase(iReadable* in)
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


tAnnLayerBase::~tAnnLayerBase()
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


void tAnnLayerBase::setAlpha(fml alpha)
{
    if (alpha <= FML(0.0))
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_alpha = alpha;
}


void tAnnLayerBase::setViscosity(fml viscosity)
{
    if (viscosity <= FML(0.0) || viscosity >= FML(1.0))
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_viscosity = viscosity;
}


fml tAnnLayerBase::calculateError(const tIO& output, const tIO& target)
{
    if (m_type == kLayerTypeSoftmax)
        return crossEntropyCost(output, target);
    else
        return standardSquaredError(output, target);
}


fml tAnnLayerBase::calculateError(const std::vector<tIO>& outputs,
                                  const std::vector<tIO>& targets)
{
    if (m_type == kLayerTypeSoftmax)
        return crossEntropyCost(outputs, targets);
    else
        return standardSquaredError(outputs, targets);
}


void tAnnLayerBase::reset()
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


void tAnnLayerBase::printLayerInfo(std::ostream& out) const
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


std::string tAnnLayerBase::layerInfoString() const
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
    // TODO
    return NULL;
}


static u32 layerId = 27424;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tAnnLayerBase::headerId() const
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


void tAnnLayerBase::pack(iWritable* out) const
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


void tAnnLayerBase::unpack(iReadable* in)
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
