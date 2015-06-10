#include <ml/tCnnLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


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


tCnnLayerBase::tCnnLayerBase()
    : m_type(kLayerTypeLogistic),
      m_rule(kWeightUpRuleNone),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_inputRows(0),
      m_inputCols(0),
      m_inputComponents(0),
      m_kernelRows(0),
      m_kernelCols(0),
      m_kernelStepY(0),
      m_kernelStepX(0),
      m_numKernels(0),
      m_outputRows(0),
      m_outputCols(0),
      m_curCount(0),
      m_maxCount(0),
      m_w(NULL),
      m_b(NULL),
      m_w_orig(NULL),
      m_b_orig(NULL)
{
}


tCnnLayerBase::tCnnLayerBase(nLayerType type, nLayerWeightUpdateRule rule,
                             u32 inputRows, u32 inputCols, u32 inputComponents,
                             u32 kernelRows, u32 kernelCols,
                             u32 kernelStepY, u32 kernelStepX,
                             u32 numKernels,
                             algo::iLCG& lcg, fml randWeightMin, fml randWeightMax)
    : m_type(type),
      m_rule(rule),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_inputRows(inputRows),
      m_inputCols(inputCols),
      m_inputComponents(inputComponents),
      m_kernelRows(kernelRows),
      m_kernelCols(kernelCols),
      m_kernelStepY(kernelStepY),
      m_kernelStepX(kernelStepX),
      m_numKernels(numKernels),
      m_outputRows(0),
      m_outputCols(0),
      m_curCount(0),
      m_maxCount(0),
      m_w(NULL),
      m_b(NULL),
      m_w_orig(NULL),
      m_b_orig(NULL)
{
    m_validate();
    m_calculateOutputSize();
    m_initWeights(lcg, randWeightMin, randWeightMax);
}


tCnnLayerBase::~tCnnLayerBase()
{
    delete [] m_w; m_w = NULL;
    delete [] m_b; m_b = NULL;
    delete [] m_w_orig; m_w_orig = NULL;
    delete [] m_b_orig; m_b_orig = NULL;
}


void tCnnLayerBase::setAlpha(fml alpha)
{
    if (alpha <= FML(0.0))
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_alpha = alpha;
}


void tCnnLayerBase::setViscosity(fml viscosity)
{
    if (viscosity <= FML(0.0) || viscosity >= FML(1.0))
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_viscosity = viscosity;
}


fml tCnnLayerBase::calculateError(const tIO& output, const tIO& target)
{
    if (m_type == kLayerTypeSoftmax)
        return crossEntropyCost(output, target);
    else
        return standardSquaredError(output, target);
}


fml tCnnLayerBase::calculateError(const std::vector<tIO>& outputs,
                                  const std::vector<tIO>& targets)
{
    if (m_type == kLayerTypeSoftmax)
        return crossEntropyCost(outputs, targets);
    else
        return standardSquaredError(outputs, targets);
}


void tCnnLayerBase::reset()
{
    if (!m_w_orig || !m_b_orig)
        throw eRuntimeError("Cannot reset this cnn layer because there is no original data. This is probably because you unpacked this layer from a stream.");

    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    for (u32 i = 0; i < numWeights; i++)
        m_w[i] = m_w_orig[i];

    u32 numBiases = m_numKernels;
    for (u32 i = 0; i < numBiases; i++)
        m_b[i] = m_b_orig[i];
}


void tCnnLayerBase::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    out << std::setw(w) << "CNN Layer:";
    out << std::setw(w) << layerTypeToString(m_type);
    out << std::setw(w) << weightUpRuleToString(m_rule);

    {
        std::ostringstream o;
        o << "a=" << m_alpha;
        if (m_rule == kWeightUpRuleMomentum)
            o << " v=" << m_viscosity;
        out << std::setw(w) << o.str();
    }

    std::ostringstream o;
    o << m_kernelRows << "x" << m_kernelCols
      << " (" << m_kernelStepY << "x" << m_kernelStepX << ")"
      << " => " << m_outputRows << "x" << m_outputCols << "x" << m_numKernels;

    out << std::setw(w) << o.str();

    out << std::endl;
}


std::string tCnnLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << m_kernelRows << 'x' << m_kernelCols;
    o << layerTypeToChar(m_type);
    o << '-';

    o << weightUpRuleToChar(m_rule);
    o << '-';

    o << "a" << m_alpha;
    if (m_rule == kWeightUpRuleMomentum)
        o << "v" << m_viscosity;

    return o.str();
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


void tCnnLayerBase::pack(iWritable* out) const
{
    rho::pack(out, (i32)m_type);
    rho::pack(out, (i32)m_rule);

    rho::pack(out, m_alpha);
    rho::pack(out, m_viscosity);

    rho::pack(out, m_inputRows);
    rho::pack(out, m_inputCols);
    rho::pack(out, m_inputComponents);
    rho::pack(out, m_kernelRows);
    rho::pack(out, m_kernelCols);
    rho::pack(out, m_kernelStepY);
    rho::pack(out, m_kernelStepX);
    rho::pack(out, m_numKernels);

    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    s_assert_writeAll(out, m_w, numWeights);

    u32 numBiases = m_numKernels;
    s_assert_writeAll(out, m_b, numBiases);
}


void tCnnLayerBase::unpack(iReadable* in)
{
    i32 type, rule;
    rho::unpack(in, type);
    rho::unpack(in, rule);
    m_type = (nLayerType)type;
    m_rule = (nLayerWeightUpdateRule)rule;

    rho::unpack(in, m_alpha);
    rho::unpack(in, m_viscosity);

    rho::unpack(in, m_inputRows);
    rho::unpack(in, m_inputCols);
    rho::unpack(in, m_inputComponents);
    rho::unpack(in, m_kernelRows);
    rho::unpack(in, m_kernelCols);
    rho::unpack(in, m_kernelStepY);
    rho::unpack(in, m_kernelStepX);
    rho::unpack(in, m_numKernels);

    m_validate();
    m_calculateOutputSize();

    m_curCount = 0;
    m_maxCount = 0;

    delete [] m_w; m_w = NULL;
    delete [] m_b; m_b = NULL;
    delete [] m_w_orig; m_w_orig = NULL;
    delete [] m_b_orig; m_b_orig = NULL;

    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    m_w = new fml[numWeights];
    s_assert_readAll(in, m_w, numWeights);

    u32 numBiases = m_numKernels;
    m_b = new fml[numBiases];
    s_assert_readAll(in, m_b, numBiases);
}


void tCnnLayerBase::m_validate()
{
    if (m_inputRows == 0 || m_inputCols == 0 || m_inputComponents == 0)
        throw eInvalidArgument("The number of input rows, cols, and components may not be zero.");
    if ((m_kernelRows % 2) == 0 || (m_kernelCols % 2) == 0)
        throw eInvalidArgument("The kernel rows and cols must both be odd.");
    if (m_kernelStepY == 0 || m_kernelStepX == 0)
        throw eInvalidArgument("The kernel step must not be zero.");
    if (m_numKernels == 0)
        throw eInvalidArgument("You cannot have zero kernels.");
}


void tCnnLayerBase::m_calculateOutputSize()
{
    m_outputRows = (m_inputRows - 1) / m_kernelStepY + 1;
    m_outputCols = (m_inputCols - 1) / m_kernelStepX + 1;
}


void tCnnLayerBase::m_initWeights(algo::iLCG& lcg,
                                  fml randWeightMin,
                                  fml randWeightMax)
{
    u32 numWeights = m_kernelRows * m_kernelCols * m_inputComponents * m_numKernels;
    delete [] m_w;
    delete [] m_w_orig;
    m_w = new fml[numWeights];
    m_w_orig = new fml[numWeights];
    for (u32 i = 0; i < numWeights; i++)
        m_w[i] = m_w_orig[i] = s_randInRange(lcg, randWeightMin, randWeightMax);

    u32 numBiases = m_numKernels;
    delete [] m_b;
    delete [] m_b_orig;
    m_b = new fml[numBiases];
    m_b_orig = new fml[numBiases];
    for (u32 i = 0; i < numBiases; i++)
        m_b[i] = m_b_orig[i] = s_randInRange(lcg, randWeightMin, randWeightMax);
}


}   // namespace ml
