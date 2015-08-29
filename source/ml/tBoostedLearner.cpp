#include <ml/tBoostedLearner.h>

#include <sstream>


namespace ml
{


tBoostedLearner::tBoostedLearner(u32 numInputDims, u32 numOutputDims)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_learners(),
      m_evaluator(NULL)
{
    if (m_numInputDims == 0)
        throw eInvalidArgument("The number of input dimensions may not be zero.");
    if (m_numOutputDims == 0)
        throw eInvalidArgument("The number of output dimensions may not be zero.");
}

tBoostedLearner::tBoostedLearner(iReadable* in)
    : m_numInputDims(0),
      m_numOutputDims(0),
      m_learners(),
      m_evaluator(NULL)
{
    this->unpack(in);
}

tBoostedLearner::~tBoostedLearner()
{
    m_numInputDims = 0;
    m_numOutputDims = 0;

    delete m_evaluator;
    m_evaluator = NULL;

    for (size_t i = 0; i < m_learners.size(); i++)
        delete m_learners[i].first;
    m_learners.clear();
}

void tBoostedLearner::addLearner(iLearner* learner, fml weight)
{
    if (learner == NULL)
        throw eInvalidArgument("The learner may not be NULL!");
    if (weight <= FML(0.0))
        throw eInvalidArgument("The weight may not be <= 0.0");
    m_learners.push_back(std::make_pair(learner, weight));
}

void tBoostedLearner::setOutputPerformanceEvaluator(iOutputPerformanceEvaluator* evaluator)
{
    delete m_evaluator;
    m_evaluator = evaluator;
}

void tBoostedLearner::addExample(const tIO& input, const tIO& target)
{
    throw eLogicError("Don't try to train a boosted learner this way! You have to train each weak learner independently.");
}

void tBoostedLearner::update()
{
    throw eLogicError("Don't try to update a boosted learner!");
}

void tBoostedLearner::evaluate(const tIO& input, tIO& output)
{
    if (m_learners.size() == 0)
        throw eLogicError("This boosted learner has no learners added to it.");
    if (input.size() != m_numInputDims)
        throw eLogicError("Unexpected input dimensionality.");

    fml weight = m_learners[0].second;
    m_learners[0].first->evaluate(input, output);
    if (output.size() != m_numOutputDims)
        throw eLogicError("Unexpected output dimensionality.");
    for (size_t j = 0; j < output.size(); j++)
        output[j] *= weight;

    if (m_learners.size() > 1)
    {
        tIO tempOutput;
        for (size_t i = 1; i < m_learners.size(); i++)
        {
            weight = m_learners[i].second;
            m_learners[i].first->evaluate(input, tempOutput);
            if (output.size() != tempOutput.size())
                throw eLogicError("Weak learner output dimensionalities do not match!");
            for (size_t j = 0; j < output.size(); j++)
                output[j] += tempOutput[j] * weight;
        }
    }
}

void tBoostedLearner::evaluateBatch(const std::vector<tIO>& inputs,
                                          std::vector<tIO>& outputs)
{
    outputs.resize(inputs.size());
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tBoostedLearner::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                    std::vector<tIO>::const_iterator inputEnd,
                                    std::vector<tIO>::iterator outputStart)
{
    if (m_learners.size() == 0)
        throw eLogicError("This boosted learner has no learners added to it.");

    if (inputEnd - inputStart <= 0)
        throw eLogicError("There are no inputs!");

    for (std::vector<tIO>::const_iterator itr = inputStart; itr != inputEnd; itr++)
    {
        if (itr->size() != m_numInputDims)
            throw eLogicError("Unexpected input dimensionality.");
    }

    fml weight = m_learners[0].second;
    m_learners[0].first->evaluateBatch(inputStart, inputEnd, outputStart);
    for (std::vector<tIO>::const_iterator itr = inputStart; itr != inputEnd; itr++)
    {
        tIO& output = *(outputStart + (itr-inputStart));
        if (output.size() != m_numOutputDims)
            throw eLogicError("Unexpected output dimensionality.");
        for (size_t j = 0; j < output.size(); j++)
            output[j] *= weight;
    }

    if (m_learners.size() > 1)
    {
        std::vector<tIO> tempOutputs(inputEnd-inputStart);
        for (size_t i = 1; i < m_learners.size(); i++)
        {
            weight = m_learners[i].second;
            m_learners[i].first->evaluateBatch(inputStart, inputEnd, tempOutputs.begin());
            for (std::vector<tIO>::const_iterator itr = inputStart; itr != inputEnd; itr++)
            {
                tIO& output = *(outputStart + (itr-inputStart));
                tIO& tempOutput = tempOutputs[itr-inputStart];
                if (output.size() != tempOutput.size())
                    throw eLogicError("Weak learner output dimensionalities do not match!");
                for (size_t j = 0; j < output.size(); j++)
                    output[j] += tempOutput[j] * weight;
            }
        }
    }
}

iOutputPerformanceEvaluator* tBoostedLearner::getOutputPerformanceEvaluator()
{
    if (!m_evaluator)
        throw eRuntimeError("The output performance evaluator object has not been set! It is NULL!");
    return m_evaluator;
}

void tBoostedLearner::reset()
{
    throw eLogicError("Don't try to reset a boosted learner!");
}

void tBoostedLearner::printLearnerInfo(std::ostream& out) const
{
    if (m_learners.size() == 0)
        throw eLogicError("This boosted learner has no learners added to it.");

    out << "Boosted learner with " << m_learners.size() << " weak learners:" << std::endl << std::endl;

    for (size_t i = 0; i < m_learners.size(); i++)
    {
        m_learners[i].first->printLearnerInfo(out);
        out << std::endl;
    }
}

std::string tBoostedLearner::learnerInfoString() const
{
    if (m_learners.size() == 0)
        throw eLogicError("This boosted learner has no learners added to it.");

    std::ostringstream out;
    out << "boosted_" << m_learners.size() << "_learners";
    return out.str();
}

static
iLearner* s_newLearnerFunc(iReadable* in)
{
    return new tBoostedLearner(in);
}

static u32 learnerId = 13973260;
static bool didRegister = iLearner::registerLearnerFuncWithHeaderId(s_newLearnerFunc, learnerId);

u32 tBoostedLearner::headerId() const
{
    if (!didRegister)
        throw eRuntimeError("Registering my learner id didn't work!");
    return learnerId;
}

void tBoostedLearner::pack(iWritable* out) const
{
    if (m_learners.size() == 0)
        throw eLogicError("This boosted learner has no learners added to it.");
    rho::pack(out, (u32)(m_learners.size()));

    for (size_t i = 0; i < m_learners.size(); i++)
    {
        rho::pack(out, m_learners[i].second);
        iLearner::writeLearnerToStream(m_learners[i].first, out);
    }

    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numOutputDims);
}

void tBoostedLearner::unpack(iReadable* in)
{
    u32 numLearners; rho::unpack(in, numLearners);
    if (numLearners == 0)
        throw eRuntimeError("Cannot unpack boosted learner with no weak learners!");

    for (size_t i = 0; i < m_learners.size(); i++)
        delete m_learners[i].first;
    m_learners.clear();
    for (u32 i = 0; i < numLearners; i++)
    {
        fml weight;  rho::unpack(in, weight);
        iLearner* learner = iLearner::newLearnerFromStream(in);
        m_learners.push_back(std::make_pair(learner, weight));
    }

    rho::unpack(in, m_numInputDims);
    if (m_numInputDims == 0)
        throw eInvalidArgument("The number of input dimensions may not be zero.");
    rho::unpack(in, m_numOutputDims);
    if (m_numOutputDims == 0)
        throw eInvalidArgument("The number of output dimensions may not be zero.");

    setOutputPerformanceEvaluator(NULL);
}


}   // namespace ml
