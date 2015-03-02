#include <ml2/tAnnLayer.h>

#include "Eigen.h"


namespace ml2
{


typedef Eigen::Matrix<fml,Eigen::Dynamic,Eigen::Dynamic> Mat;
typedef Eigen::Map<Mat> Map;


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


tAnnLayer::tAnnLayer(nAnnLayerType type, nAnnLayerWeightUpdateRule rule)
    : m_type(type),
      m_rule(rule),
      m_alpha(FML(0.0)),
      m_viscosity(FML(0.0)),
      m_output(NULL),
      m_numOutputDims(0),
      m_outputCount(0),
      m_prev_da(NULL),
      m_numInputDims(0),
      m_inputCount(0),
      m_weights(NULL),
      m_A(NULL),
      m_a(NULL)
{
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
    if (m_numInputDims != numInputDims)
    {
        if (m_numInputDims == 0)
            m_numInputDims = numInputDims;
        else
            throw eInvalidArgument("Unexpected numInputDims");
    }

    Map inputMap(input, numInputDims, count);
    Map weights(m_weights, 1, m_numInputDims);
    Map A(m_A, 1, count);
    Map a(m_a, 1, count);

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
