#include <ml2/tLayeredLearnerBase.h>

#include <iomanip>
#include <sstream>


namespace ml2
{


tLayeredLearnerBase::tLayeredLearnerBase(u32 numInputDims, u32 numOutputDims)
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

tLayeredLearnerBase::tLayeredLearnerBase(iReadable* in)
    : m_layers(),
      m_numInputDims(0),
      m_numOutputDims(0),
      m_inputMatrix(NULL),
      m_inputMatrixSize(0),
      m_inputMatrixUsed(0),
      m_targetMatrix(NULL),
      m_targetMatrixSize(0),
      m_targetMatrixUsed(0)
{
    this->unpack(in);
}

tLayeredLearnerBase::~tLayeredLearnerBase()
{
    delete [] m_inputMatrix;
    m_inputMatrix = NULL;

    delete [] m_targetMatrix;
    m_targetMatrix = NULL;

    for (size_t i = 0; i < m_layers.size(); i++)
        delete m_layers[i];
    m_layers.clear();
}

void tLayeredLearnerBase::addLayer(iLayer* layer)
{
    m_layers.push_back(layer);
}

void tLayeredLearnerBase::addExample(const tIO& input, const tIO& target)
{
    m_copyToInputMatrix(input);
    m_copyToTargetMatrix(target);
}

fml tLayeredLearnerBase::calculateError(const tIO& output, const tIO& target)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot calculate error when there are no layers!");
    return m_layers.back()->calculateError(output, target);
}

fml tLayeredLearnerBase::calculateError(const std::vector<tIO>& outputs,
                                        const std::vector<tIO>& targets)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot calculate error when there are no layers!");
    return m_layers.back()->calculateError(outputs, targets);
}

void tLayeredLearnerBase::reset()
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot reset when there are no layers!");
    for (size_t i = 0; i < m_layers.size(); i++)
        m_layers[i]->reset();
    m_clearMatrices();
}

void tLayeredLearnerBase::printLearnerInfo(std::ostream& out) const
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot print learner info when there are no layers!");

    int w = 25;

    out << std::setw(w) << " ";
    out << std::setw(w) << "Process Type:";
    out << std::setw(w) << "Train Rule:";
    out << std::setw(w) << "Train Parameters:";
    out << std::setw(w) << "# Output Dimensions:";
    out << std::endl;

    out << std::setw(w) << "Input Layer:";
    out << std::setw(w) << "-";
    out << std::setw(w) << "-";
    out << std::setw(w) << "-";
    out << std::setw(w) << m_numInputDims;
    out << std::endl;

    for (size_t i = 0; i < m_layers.size(); i++)
        m_layers[i]->printLayerInfo(out);
    out << std::endl;
}

std::string tLayeredLearnerBase::learnerInfoString() const
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot get learner info string when there are no layers!");
    std::ostringstream out;
    out << m_numInputDims;
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        out << "__";
        out << m_layers[i]->layerInfoString();
    }
    return out.str();
}

void tLayeredLearnerBase::pack(iWritable* out) const
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot pack when there are no layers!");
    rho::pack(out, (u32)(m_layers.size()));

    for (size_t i = 0; i < m_layers.size(); i++)
        iLayer::writeLayerToStream(m_layers[i], out);

    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numOutputDims);
}

void tLayeredLearnerBase::unpack(iReadable* in)
{
    u32 numLayers; rho::unpack(in, numLayers);
    if (numLayers == 0)
        throw eRuntimeError("Cannot unpack layered learner with no layers!");

    for (size_t i = 0; i < m_layers.size(); i++)
        delete m_layers[i];
    m_layers.clear();
    for (u32 i = 0; i < numLayers; i++)
        m_layers.push_back(iLayer::newLayerFromStream(in));

    rho::unpack(in, m_numInputDims);
    if (m_numInputDims == 0)
        throw eInvalidArgument("The number of input dimensions may not be zero.");
    rho::unpack(in, m_numOutputDims);
    if (m_numOutputDims == 0)
        throw eInvalidArgument("The number of output dimensions may not be zero.");

    delete [] m_inputMatrix;
    m_inputMatrixSize = 1024;
    m_inputMatrix = new fml[m_inputMatrixSize];

    delete [] m_targetMatrix;
    m_targetMatrixSize = 1024;
    m_targetMatrix = new fml[m_targetMatrixSize];

    m_clearMatrices();
}

void tLayeredLearnerBase::m_growInputMatrix(u32 newSize)
{
    while (newSize > m_inputMatrixSize)
    {
        fml* inputMatrix = new fml[2*m_inputMatrixSize];
        memcpy(inputMatrix, m_inputMatrix, m_inputMatrixSize*sizeof(fml));
        delete [] m_inputMatrix;
        m_inputMatrix = inputMatrix;
        m_inputMatrixSize *= 2;
    }
}

void tLayeredLearnerBase::m_growTargetMatrix(u32 newSize)
{
    while (newSize > m_targetMatrixSize)
    {
        fml* targetMatrix = new fml[2*m_targetMatrixSize];
        memcpy(targetMatrix, m_targetMatrix, m_targetMatrixSize*sizeof(fml));
        delete [] m_targetMatrix;
        m_targetMatrix = targetMatrix;
        m_targetMatrixSize *= 2;
    }
}

void tLayeredLearnerBase::m_copyToInputMatrix(const tIO& input)
{
    if ((u32)input.size() != m_numInputDims)
        throw eInvalidArgument("Unexpected input dimensionality.");
    u32 newUsed = m_inputMatrixUsed + (u32)input.size();
    m_growInputMatrix(newUsed);
    fml* inputMatrix = m_inputMatrix + m_inputMatrixUsed;
    memcpy(inputMatrix, &(input[0]), input.size()*sizeof(fml));
    m_inputMatrixUsed = newUsed;
}

void tLayeredLearnerBase::m_copyToTargetMatrix(const tIO& target)
{
    if ((u32)target.size() != m_numOutputDims)
        throw eInvalidArgument("Unexpected target dimensionality.");
    u32 newUsed = m_targetMatrixUsed + (u32)target.size();
    m_growTargetMatrix(newUsed);
    fml* targetMatrix = m_targetMatrix + m_targetMatrixUsed;
    memcpy(targetMatrix, &(target[0]), target.size()*sizeof(fml));
    m_targetMatrixUsed = newUsed;
}

void tLayeredLearnerBase::m_clearMatrices()
{
    m_inputMatrixUsed = 0;
    m_targetMatrixUsed = 0;
}

void tLayeredLearnerBase::m_pushInputForward(const fml* input, u32 numInputDims, u32 inputCount,
                                             const fml*& output, u32 expectedOutputDims, u32 expectedOutputCount)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot push input when there are no layers!");

    m_layers[0]->takeInput(input, numInputDims,
                           inputCount);

    u32 prevOutDims, prevCount;
    const fml* prevOutput = NULL;

    for (size_t i = 1; i < m_layers.size(); i++)
    {
        prevOutput = m_layers[i-1]->getOutput(prevOutDims, prevCount);
        m_layers[i]->takeInput(prevOutput, prevOutDims, prevCount);
    }

    prevOutput = m_layers.back()->getOutput(prevOutDims, prevCount);

    if (prevOutDims != expectedOutputDims)
        throw eRuntimeError("Unexpected last layer dimensionality!!!");
    if (prevCount != expectedOutputCount)
        throw eRuntimeError("Unexpected last layer count!!!");

    output = prevOutput;
}

void tLayeredLearnerBase::m_backpropagate(const fml* output_da, u32 numOutputDims, u32 outputCount,
                                          const fml* input, u32 numInputDims, u32 inputCount)
{
    if (m_layers.size() == 0)
        throw eRuntimeError("Cannot backpropagate when there are no layers!");

    for (size_t i = (m_layers.size()-1); i > 0; i--)
    {
        u32 prevOutDims, prevCount;
        const fml* prevOutput = m_layers[i-1]->getOutput(prevOutDims, prevCount);
        m_layers[i]->takeOutputErrorGradients(output_da, numOutputDims, outputCount,
                                              prevOutput, prevOutDims, prevCount,
                                              true);
        output_da = m_layers[i]->getInputErrorGradients(numOutputDims, outputCount);
    }

    m_layers[0]->takeOutputErrorGradients(output_da, numOutputDims, outputCount,
                                          input, numInputDims, inputCount,
                                          false);
}

void tLayeredLearnerBase::m_putOutput(tIO& output, const fml* outputPtr, u32 numOutputDims, u32 outputCount)
{
    if (outputCount != 1)
        throw eRuntimeError("Unexpected output count");
    output.resize(numOutputDims);
    memcpy(&(output[0]), outputPtr, numOutputDims*sizeof(fml));
}

void tLayeredLearnerBase::m_putOutput(std::vector<tIO>::iterator outputStart, const fml* outputPtr, u32 numOutputDims, u32 outputCount)
{
    std::vector<tIO>::iterator oitr = outputStart;
    for (u32 j = 0; j < outputCount; j++)
    {
        oitr->resize(numOutputDims);
        memcpy(&((*oitr)[0]), outputPtr, numOutputDims*sizeof(fml));
        outputPtr += numOutputDims;
        oitr++;
    }
}

void tLayeredLearnerBase::m_update(const fml* inputMatrix, u32 inputMatrixUsed, u32 inputMatrixNumDims,
                                         fml* targetMatrix, u32 targetMatrixUsed, u32 targetMatrixNumDims)
{
    if (inputMatrixUsed == 0)
        throw eRuntimeError("Cannot update when there have been no example inputs.");
    if ((inputMatrixUsed % inputMatrixNumDims) != 0)
        throw eRuntimeError("Wack inputMatrixUsed");

    const fml* input = inputMatrix;
    u32 numInputDims = inputMatrixNumDims;
    u32 inputCount = inputMatrixUsed / inputMatrixNumDims;

    const fml* output = NULL;
    u32 expectedOutputDims = targetMatrixNumDims;
    u32 expectedOutputCount = inputCount;

    m_pushInputForward(input, numInputDims, inputCount,
                       output, expectedOutputDims, expectedOutputCount);

    if (targetMatrixUsed == 0)
        throw eRuntimeError("Cannot update when there have been no example targets.");
    if ((targetMatrixUsed % targetMatrixNumDims) != 0)
        throw eRuntimeError("Wack targetMatrixUsed");
    if ((targetMatrixUsed / targetMatrixNumDims) != expectedOutputCount)
        throw eRuntimeError("Wack targetMatrixUsed (2)");

    m_calculate_output_da(output, targetMatrix, expectedOutputDims, expectedOutputCount);

    const fml* output_da = targetMatrix;

    m_backpropagate(output_da, expectedOutputDims, expectedOutputCount,
                    input, numInputDims, inputCount);
}

void tLayeredLearnerBase::m_evaluate(const fml* inputMatrix, u32 inputMatrixUsed, u32 inputMatrixNumDims,
                                     const fml*& output, u32 expectedOutputDims, u32 expectedOutputCount)
{
    if (inputMatrixUsed == 0)
        throw eRuntimeError("Cannot update when there have been no example inputs.");
    if ((inputMatrixUsed % inputMatrixNumDims) != 0)
        throw eRuntimeError("Wack inputMatrixUsed");

    const fml* inputPtr = inputMatrix;
    u32 numInputDims = inputMatrixNumDims;
    u32 inputCount = inputMatrixUsed / inputMatrixNumDims;

    m_pushInputForward(inputPtr, numInputDims, inputCount,
                       output, expectedOutputDims, expectedOutputCount);
}


}   // namespace ml2
