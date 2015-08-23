#include <ml/tScalingLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


tScalingLayerBase::tScalingLayerBase()
    : m_numInputDims(0),
      m_numOutputDims(0),
      m_scaleFactor(FML(0.0)),
      m_curCount(0),
      m_maxCount(0)
{
}

tScalingLayerBase::tScalingLayerBase(u32 numInputDims, u32 numOutputDims, fml scaleFactor)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_scaleFactor(scaleFactor),
      m_curCount(0),
      m_maxCount(0)
{
    if (numInputDims == 0)
        throw eInvalidArgument("numInputDims must be positive!");
    if (numOutputDims == 0)
        throw eInvalidArgument("numOutputDims must be positive!");
}

tScalingLayerBase::~tScalingLayerBase()
{
    // Nothing needed here...
}

void tScalingLayerBase::reset()
{
    // Nothing needed here...
}

void tScalingLayerBase::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    // Class Name:
    out << std::setw(w) << "Scaling Layer:";

    // Process Type:
    out << std::setw(w) << '-';

    // Train Rule:
    out << std::setw(w) << '-';

    // Train Parameters:
    out << std::setw(w) << '-';

    // Layer Parameters:
    {
        std::ostringstream o;
        o << "* " << m_scaleFactor;
        out << std::setw(w) << o.str();
    }

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

std::string tScalingLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << m_scaleFactor << "xS";

    return o.str();
}

void tScalingLayerBase::pack(iWritable* out) const
{
    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numOutputDims);
    rho::pack(out, m_scaleFactor);
}

void tScalingLayerBase::unpack(iReadable* in)
{
    rho::unpack(in, m_numInputDims);
    rho::unpack(in, m_numOutputDims);
    rho::unpack(in, m_scaleFactor);
    m_curCount = 0;
    m_maxCount = 0;
}


}   // namespace ml
