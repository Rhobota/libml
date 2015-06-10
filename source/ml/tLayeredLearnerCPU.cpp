#include <ml/tLayeredLearnerCPU.h>

#include "Eigen.h"


namespace ml
{


tLayeredLearnerCPU::tLayeredLearnerCPU(u32 numInputDims, u32 numOutputDims)
    : tLayeredLearnerBase(numInputDims, numOutputDims)
{
}

tLayeredLearnerCPU::tLayeredLearnerCPU(iReadable* in)
    : tLayeredLearnerBase(in)
{
}

tLayeredLearnerCPU::~tLayeredLearnerCPU()
{
    // The super d'tors are called automatically.
}

void tLayeredLearnerCPU::update()
{
    m_update(m_inputMatrix, m_inputMatrixUsed, m_numInputDims,
             m_targetMatrix, m_targetMatrixUsed, m_numOutputDims);
    m_clearMatrices();
}

void tLayeredLearnerCPU::evaluate(const tIO& input, tIO& output)
{
    m_clearMatrices();
    m_copyToInputMatrix(input);

    const fml* outputPtr = NULL;
    u32 expectedOutputDims = m_numOutputDims;
    u32 expectedOutputCount = 1;

    m_evaluate(m_inputMatrix, m_inputMatrixUsed, m_numInputDims,
               outputPtr, expectedOutputDims, expectedOutputCount);
    m_putOutput(output, outputPtr, expectedOutputDims, expectedOutputCount);

    m_clearMatrices();
}

void tLayeredLearnerCPU::evaluateBatch(const std::vector<tIO>& inputs,
                                             std::vector<tIO>& outputs)
{
    outputs.resize(inputs.size());
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tLayeredLearnerCPU::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                       std::vector<tIO>::const_iterator inputEnd,
                                       std::vector<tIO>::iterator outputStart)
{
    m_clearMatrices();
    std::vector<tIO>::const_iterator sitr;
    for (sitr = inputStart; sitr != inputEnd; sitr++)
        m_copyToInputMatrix(*sitr);

    const fml* outputPtr = NULL;
    u32 expectedOutputDims = m_numOutputDims;
    u32 expectedOutputCount = (u32) (inputEnd - inputStart);

    m_evaluate(m_inputMatrix, m_inputMatrixUsed, m_numInputDims,
               outputPtr, expectedOutputDims, expectedOutputCount);
    m_putOutput(outputStart, outputPtr, expectedOutputDims, expectedOutputCount);

    m_clearMatrices();
}

static
iLearner* s_newLearnerFunc(iReadable* in)
{
    return new tLayeredLearnerCPU(in);
}

static u32 learnerId = 2742490;
static bool didRegister = iLearner::registerLearnerFuncWithHeaderId(s_newLearnerFunc, learnerId);

u32 tLayeredLearnerCPU::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my learner id didn't work!");
    return learnerId;
}

void tLayeredLearnerCPU::m_calculate_output_da(const fml* output, fml* target, u32 dims, u32 count)
{
    MapConst a(output, dims, count);
    Map y(target, dims, count);
    y = a - y;
}


}   // namespace ml
