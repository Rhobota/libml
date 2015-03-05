#include <ml2/tLayeredLearnerGPU.h>


namespace ml2
{


tLayeredLearnerGPU::tLayeredLearnerGPU(u32 numInputDims, u32 numOutputDims)
    : tLayeredLearnerBase(numInputDims, numOutputDims)
{
}

tLayeredLearnerGPU::tLayeredLearnerGPU(iReadable* in)
    : tLayeredLearnerBase(in)
{
}

tLayeredLearnerGPU::~tLayeredLearnerGPU()
{
    // The super d'tors are called automatically.
}

void tLayeredLearnerGPU::update()
{
    // TODO
}

void tLayeredLearnerGPU::evaluate(const tIO& input, tIO& output)
{
    // TODO
}

void tLayeredLearnerGPU::evaluateBatch(const std::vector<tIO>& inputs,
                                             std::vector<tIO>& outputs)
{
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tLayeredLearnerGPU::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                       std::vector<tIO>::const_iterator inputEnd,
                                       std::vector<tIO>::iterator outputStart)
{
    // TODO
}

static
iLearner* s_newLearnerFunc(iReadable* in)
{
    return new tLayeredLearnerGPU(in);
}

static u32 learnerId = 87986;
static bool didRegister = iLearner::registerLearnerFuncWithHeaderId(s_newLearnerFunc, learnerId);

u32 tLayeredLearnerGPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my learner id didn't work!");
    return learnerId;
}

void tLayeredLearnerGPU::m_calculate_output_da(const fml* output, fml* target, u32 dims, u32 count)
{
    // TODO
}


}   // namespace ml2
