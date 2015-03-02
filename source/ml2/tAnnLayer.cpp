#include <ml2/tAnnLayer.h>

#include "Eigen.h"


namespace ml2
{


typedef Eigen::Matrix< fml, Eigen::Dynamic, Eigen::Dynamic > Mat;
typedef Eigen::Map< Mat, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic> > Map;


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


class tHyperbolicFunc
{
    public:

        fml operator()(fml val) const { return hyperbolic_function(val); }
};


tAnnLayer::tAnnLayer(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                     u32 numNeurons)
    : m_type(type),
      m_rule(rule),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_numInputDims(0),
      m_numNeurons(numNeurons),
      m_weights(NULL),
      m_curCount(0),
      m_maxCount(0),
      m_A(NULL),
      m_a(NULL),
      m_prev_da(NULL)
{
    if (m_numNeurons == 0)
        throw eInvalidArgument("The number of neurons may not be zero.");
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


void tAnnLayer::takeInput(fml* input, u32 numInputDims, u32 count)
{
    if (!input)
        throw eInvalidArgument("The input matrix may not be null.");

    if (m_numInputDims != numInputDims || numInputDims == 0)
    {
        if (m_numInputDims == 0 && numInputDims > 0)
            m_numInputDims = numInputDims;
        else
            throw eInvalidArgument("Unexpected numInputDims!");
    }

    if (count == 0)
        throw eInvalidArgument("The count may not be zero.");

    if (!m_weights)
    {
        m_weights = new fml[m_numNeurons * numInputDims];
        // TODO -- randomize weights here.
    }

    if (!m_A || !m_a || count > m_maxCount)
    {
        delete [] m_A;
        delete [] m_a;
        m_A = new fml[m_numNeurons * count];
        m_a = new fml[(m_numNeurons+1) * count];
        m_maxCount = count;
        Map a(m_a, (m_numNeurons+1), count);
        a.bottomRows(1).setOnes();
    }
    m_curCount = count;

    Map inputMap(input, numInputDims, count);
    Map weights(m_weights, m_numNeurons, numInputDims);
    Map A(m_A, m_numNeurons, count);
    Map a(m_a, m_numNeurons, count, Eigen::OuterStride<>(m_numNeurons+1));

    A = (weights * inputMap) / ((fml) numInputDims);

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


fml* tAnnLayer::getOutput(u32& numOutputDims, u32& count) const
{
    // TODO
    return NULL;
}


void tAnnLayer::takeOutputErrorGradients(
                  fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  fml* input, u32 numInputDims, u32 inputCount)
{
    // TODO
}


fml* tAnnLayer::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    // TODO
    return NULL;
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
