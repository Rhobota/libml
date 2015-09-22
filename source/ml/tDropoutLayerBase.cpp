#include <ml/tDropoutLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


tDropoutLayerBase::tDropoutLayerBase()
    : m_numInputDims(0),
      m_numOutputDims(0),
      m_p(FML(0.0)),
      m_trainMode(false),
      m_curCount(0),
      m_maxCount(0)
{
}

tDropoutLayerBase::tDropoutLayerBase(u32 numInputDims, u32 numOutputDims, fml p)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_p(p),
      m_trainMode(false),
      m_curCount(0),
      m_maxCount(0)
{
    if (numInputDims == 0)
        throw eInvalidArgument("numInputDims must be positive!");
    if (numOutputDims == 0)
        throw eInvalidArgument("numOutputDims must be positive!");
    if (m_p <= FML(0.0) || m_p > FML(1.0))
        throw eInvalidArgument("p must be in (0,1]");
}

tDropoutLayerBase::~tDropoutLayerBase()
{
    // Nothing needed here...
}

void tDropoutLayerBase::reset()
{
    m_trainMode = false;
}

void tDropoutLayerBase::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    // Class Name:
    out << std::setw(w) << "Dropout Layer:";

    // Process Type:
    out << std::setw(w) << '-';

    // Train Rule:
    out << std::setw(w) << '-';

    // Train Parameters:
    out << std::setw(w) << '-';

    // Layer Parameters:
    {
        std::ostringstream o;
        o << "prob: " << m_p;
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

std::string tDropoutLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << m_p << "dp";

    return o.str();
}

void tDropoutLayerBase::pack(iWritable* out) const
{
    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numOutputDims);
    rho::pack(out, m_p);
}

void tDropoutLayerBase::unpack(iReadable* in)
{
    rho::unpack(in, m_numInputDims);
    rho::unpack(in, m_numOutputDims);
    rho::unpack(in, m_p);
    m_curCount = 0;
    m_maxCount = 0;
    m_trainMode = false;
}


}   // namespace ml
