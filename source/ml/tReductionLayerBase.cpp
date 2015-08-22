#include <ml/tReductionLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


tReductionLayerBase::tReductionLayerBase()
    : m_numInputDims(0),
      m_numOutputDims(0),
      m_curCount(0),
      m_maxCount(0)
{
}

tReductionLayerBase::tReductionLayerBase(u32 numInputDims, u32 numOutputDims)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_curCount(0),
      m_maxCount(0)
{
    if (numInputDims == 0)
        throw eInvalidArgument("numInputDims must be positive!");
    if (numOutputDims == 0)
        throw eInvalidArgument("numOutputDims must be positive!");
}

tReductionLayerBase::~tReductionLayerBase()
{
    // Nothing needed here...
}

fml tReductionLayerBase::calculateError(const tIO& output, const tIO& target)
{
    throw eImpossiblePath();
}

fml tReductionLayerBase::calculateError(const std::vector<tIO>& outputs,
                                      const std::vector<tIO>& targets)
{
    throw eImpossiblePath();
}

void tReductionLayerBase::reset()
{
    // Nothing needed here...
}

void tReductionLayerBase::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    // Class Name:
    out << std::setw(w) << "Reduction Layer:";

    // Process Type:
    out << std::setw(w) << '-';

    // Train Rule:
    out << std::setw(w) << '-';

    // Train Parameters:
    out << std::setw(w) << '-';

    // Layer Parameters:
    out << std::setw(w) << '-';

    // # Free Parameters:
    out << std::setw(w) << '-';

    // # Connections:
    out << std::setw(w) << '-';

    // # Output Dimensions:
    {
        std::ostringstream o;
        o << m_numOutputDims;
        out << std::setw(w) << o.str();
    }

    out << std::endl;
}

std::string tReductionLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << "rdctn";

    return o.str();
}

void tReductionLayerBase::pack(iWritable* out) const
{
    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numOutputDims);
}

void tReductionLayerBase::unpack(iReadable* in)
{
    rho::unpack(in, m_numInputDims);
    rho::unpack(in, m_numOutputDims);
    m_curCount = 0;
    m_maxCount = 0;
}


}   // namespace ml
