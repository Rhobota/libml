#include <ml2/tLayeredLearner.h>

#include <cassert>

#include "Eigen.h"


namespace ml2
{


tLayeredLearner::tLayeredLearner(u32 numInputDims, u32 numOutputDims)
    : m_layers(),
      m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_inputMatrix(NULL),
      m_inputMatrixSize(0),
      m_inputMatrixUsed(0),
      m_targetMatrix(NULL),
      m_targetMatrixSize(0),
      m_targetMatrixUsed(0)
{
    if (m_numInputDims == 0)
        throw eInvalidArgument("The number of input dimensions may not be zero.");
    if (m_numOutputDims == 0)
        throw eInvalidArgument("The number of output dimensions may not be zero.");

    m_inputMatrixSize = 1024;
    m_inputMatrix = new fml[m_inputMatrixSize];

    m_targetMatrixSize = 1024;
    m_targetMatrix = new fml[m_targetMatrixSize];
}

tLayeredLearner::~tLayeredLearner()
{
    delete [] m_inputMatrix;
    m_inputMatrix = NULL;

    delete [] m_targetMatrix;
    m_targetMatrix = NULL;
}

void tLayeredLearner::addLayer(iLayer* layer)
{
    m_layers.push_back(layer);
}

void tLayeredLearner::addExample(const tIO& input, const tIO& target)
{
    m_copyToInputMatrix(input);
    m_copyToTargetMatrix(target);
}

void tLayeredLearner::update()
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot update when there are no layers!");
    if (m_inputMatrixUsed == 0)
        throw eRuntimeError("Cannot update when there have been no example inputs.");
    if (m_targetMatrixUsed == 0)
        throw eRuntimeError("Cannot update when there have been no example targets.");
    if ((m_inputMatrixUsed % m_numInputDims) != 0)
        throw eRuntimeError("Wack m_inputMatrixUsed");
    if ((m_targetMatrixUsed % m_numOutputDims) != 0)
        throw eRuntimeError("Wack m_targetMatrixUsed");
    u32 correctCount = m_inputMatrixUsed / m_numInputDims;
    if ((m_targetMatrixUsed / m_numOutputDims) != correctCount)
        throw eRuntimeError("Wack m_targetMatrixUsed (2)");

    m_layers[0]->takeInput(m_inputMatrix, m_numInputDims,
                           correctCount);

    u32 prevOutDims, prevCount;
    const fml* prevOutput = NULL;

    for (size_t i = 1; i < m_layers.size(); i++)
    {
        prevOutput = m_layers[i-1]->getOutput(prevOutDims, prevCount);
        m_layers[i]->takeInput(prevOutput, prevOutDims, prevCount);
    }

    {
        prevOutput = m_layers.back()->getOutput(prevOutDims, prevCount);
        if (prevOutDims != m_numOutputDims)
            throw eRuntimeError("Unexpected last layer dimensionality!!!");
        if (prevCount != correctCount)
            throw eRuntimeError("Unexpected last layer count!!!");
        MapConst a(prevOutput, prevOutDims, prevCount);
        Map y(m_targetMatrix, prevOutDims, prevCount);
        y = a - y;
    }

    const fml* output_da = m_targetMatrix;
    u32 outDims = prevOutDims;
    u32 count = prevCount;

    for (size_t i = (m_layers.size()-1); i > 0; i--)
    {
        prevOutput = m_layers[i-1]->getOutput(prevOutDims, prevCount);
        m_layers[i]->takeOutputErrorGradients(output_da, outDims, count,
                                              prevOutput, prevOutDims, prevCount,
                                              true);
        output_da = m_layers[i]->getInputErrorGradients(outDims, count);
    }

    m_layers[0]->takeOutputErrorGradients(output_da, outDims, count,
                                          m_inputMatrix, m_numInputDims, correctCount,
                                          false);

    m_inputMatrixUsed = 0;
    m_targetMatrixUsed = 0;
}

void tLayeredLearner::evaluate(const tIO& input, tIO& output)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot evaluate when there are no layers!");

    m_inputMatrixUsed = 0;
    m_targetMatrixUsed = 0;
    m_copyToInputMatrix(input);

    if (m_inputMatrixUsed != m_numInputDims)
        throw eRuntimeError("Wack m_inputMatrixUsed");

    m_layers[0]->takeInput(m_inputMatrix, m_numInputDims, 1);
    m_inputMatrixUsed = 0;

    u32 prevOutDims, prevCount;
    const fml* prevOutput = NULL;

    for (size_t i = 1; i < m_layers.size(); i++)
    {
        prevOutput = m_layers[i-1]->getOutput(prevOutDims, prevCount);
        m_layers[i]->takeInput(prevOutput, prevOutDims, prevCount);
    }

    prevOutput = m_layers.back()->getOutput(prevOutDims, prevCount);
    if (prevOutDims != m_numOutputDims)
        throw eRuntimeError("Unexpected last layer dimensionality!!!");
    if (prevCount != 1)
        throw eRuntimeError("Unexpected layer layer count!!!");

    output.resize(prevOutDims);
    for (u32 i = 0; i < prevOutDims; i++)
        output[i] = prevOutput[i];
}

void tLayeredLearner::evaluateBatch(const std::vector<tIO>& inputs,
                                          std::vector<tIO>& outputs)
{
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tLayeredLearner::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                    std::vector<tIO>::const_iterator inputEnd,
                                    std::vector<tIO>::iterator outputStart)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot evaluate when there are no layers!");

    m_inputMatrixUsed = 0;
    m_targetMatrixUsed = 0;

    std::vector<tIO>::const_iterator sitr;
    for (sitr = inputStart; sitr != inputEnd; sitr++)
        m_copyToInputMatrix(*sitr);

    if (m_inputMatrixUsed == 0)
        throw eRuntimeError("Wack m_inputMatrixUsed");
    if ((m_inputMatrixUsed % m_numInputDims) != 0)
        throw eRuntimeError("Wack m_inputMatrixUsed");

    m_layers[0]->takeInput(m_inputMatrix, m_numInputDims,
                           m_inputMatrixUsed / m_numInputDims);
    m_inputMatrixUsed = 0;

    u32 prevOutDims, prevCount;
    const fml* prevOutput = NULL;

    for (size_t i = 1; i < m_layers.size(); i++)
    {
        prevOutput = m_layers[i-1]->getOutput(prevOutDims, prevCount);
        m_layers[i]->takeInput(prevOutput, prevOutDims, prevCount);
    }

    prevOutput = m_layers.back()->getOutput(prevOutDims, prevCount);

    if (prevOutDims != m_numOutputDims)
        throw eRuntimeError("Unexpected last layer dimensionality!!!");
    if (prevCount != (inputEnd-inputStart))
        throw eRuntimeError("Unexpected layer layer count!!!");

    std::vector<tIO>::iterator oitr = outputStart;
    for (u32 j = 0; j < prevCount; j++)
    {
        oitr->resize(prevOutDims);
        for (u32 i = 0; i < prevOutDims; i++)
            (*oitr)[i] = *prevOutput++;
        oitr++;
    }
}

fml tLayeredLearner::calculateError(const tIO& output, const tIO& target)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot calculate error when there are no layers!");
    return m_layers.back()->calculateError(output, target);
}

fml tLayeredLearner::calculateError(const std::vector<tIO>& outputs,
                                    const std::vector<tIO>& targets)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot calculate error when there are no layers!");
    return m_layers.back()->calculateError(outputs, targets);
}

void tLayeredLearner::reset()
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot reset when there are no layers!");
    for (size_t i = 0; i < m_layers.size(); i++)
        m_layers[i]->reset();
}

void tLayeredLearner::printLearnerInfo(std::ostream& out) const
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot print learner info when there are no layers!");
    for (size_t i = 0; i < m_layers.size(); i++)
        m_layers[i]->printLayerInfo(out);
}

std::string tLayeredLearner::learnerInfoString() const
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot get learner info string when there are no layers!");
    std::string str;
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        if (i > 0)
            str += "__";
        str += m_layers[i]->layerInfoString();
    }
    return str;
}

void tLayeredLearner::m_growInputMatrix(u32 newSize)
{
    while (newSize >= m_inputMatrixSize)
    {
        fml* inputMatrix = new fml[2*m_inputMatrixSize];
        for (u32 i = 0; i < m_inputMatrixSize; i++)
            inputMatrix[i] = m_inputMatrix[i];
        delete [] m_inputMatrix;
        m_inputMatrix = inputMatrix;
        m_inputMatrixSize *= 2;
    }
}

void tLayeredLearner::m_growTargetMatrix(u32 newSize)
{
    while (newSize >= m_targetMatrixSize)
    {
        fml* targetMatrix = new fml[2*m_targetMatrixSize];
        for (u32 i = 0; i < m_targetMatrixSize; i++)
            targetMatrix[i] = m_targetMatrix[i];
        delete [] m_targetMatrix;
        m_targetMatrix = targetMatrix;
        m_targetMatrixSize *= 2;
    }
}

void tLayeredLearner::m_copyToInputMatrix(const tIO& input)
{
    if ((u32)input.size() != m_numInputDims)
        throw eInvalidArgument("Unexpected input dimensionality.");
    u32 newUsed = m_inputMatrixUsed + (u32)input.size();
    m_growInputMatrix(newUsed);
    fml* inputMatrix = m_inputMatrix + m_inputMatrixUsed;
    for (size_t i = 0; i < input.size(); i++)
        *inputMatrix++ = input[i];
    m_inputMatrixUsed = newUsed;
}

void tLayeredLearner::m_copyToTargetMatrix(const tIO& target)
{
    if ((u32)target.size() != m_numOutputDims)
        throw eInvalidArgument("Unexpected target dimensionality.");
    u32 newUsed = m_targetMatrixUsed + (u32)target.size();
    m_growTargetMatrix(newUsed);
    fml* targetMatrix = m_targetMatrix + m_targetMatrixUsed;
    for (size_t i = 0; i < target.size(); i++)
        *targetMatrix++ = target[i];
    m_targetMatrixUsed = newUsed;
}


}   // namespace ml2
