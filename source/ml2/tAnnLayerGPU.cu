#include <ml2/tAnnLayerGPU.h>


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


class t_RMSPROP_update
{
    public:

        fml operator()(fml accum, fml accum_avg) const
        {
            return (accum_avg > FML(0.0)) ? (accum / std::sqrt(accum_avg)) : FML(0.0);
        }
};


tAnnLayerGPU::tAnnLayerGPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                           u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                           fml randWeightMin, fml randWeightMax)
    : tAnnLayerBase(type, rule, numInputDims, numNeurons, lcg,
                    randWeightMin, randWeightMax)
{
}


tAnnLayerGPU::tAnnLayerGPU(iReadable* in)
    : tAnnLayerBase(in)
{
}


tAnnLayerGPU::~tAnnLayerGPU()
{
    // The super d'tor are called automatically.
}


void tAnnLayerGPU::takeInput(const fml* input, u32 numInputDims, u32 count)
{
    // TODO
}


const fml* tAnnLayerGPU::getOutput(u32& numOutputDims, u32& count) const
{
    // TODO
    return NULL;
}


void tAnnLayerGPU::takeOutputErrorGradients(
                  const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                  const fml* input, u32 numInputDims, u32 inputCount,
                  bool calculateInputErrorGradients)
{
    // TODO
}


const fml* tAnnLayerGPU::getInputErrorGradients(u32& numInputDims, u32& count) const
{
    // TODO
    return NULL;
}


static
iLayer* s_newLayerFunc(iReadable* in)
{
    return new tAnnLayerGPU(in);
}


static u32 layerId = 78879;
static bool didRegister = iLayer::registerLayerFuncWithHeaderId(s_newLayerFunc, layerId);


u32 tAnnLayerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my layer id didn't work!");
    return layerId;
}


}   // namespace ml2
