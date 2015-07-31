#include <ml/tPoolingLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


tPoolingLayerBase::tPoolingLayerBase(u32 inputRows, u32 inputCols, u32 poolRows, u32 poolCols)
    : m_inputRows(inputRows),
      m_inputCols(inputCols),
      m_poolRows(poolRows),
      m_poolCols(poolCols),
      m_curCount(0),
      m_maxCount(0)
{
}

tPoolingLayerBase::~tPoolingLayerBase()
{
    // Nothing needed here...
}

fml tPoolingLayerBase::calculateError(const tIO& output, const tIO& target)
{
    throw eImpossiblePath();
}

fml tPoolingLayerBase::calculateError(const std::vector<tIO>& outputs,
                                      const std::vector<tIO>& targets)
{
    throw eImpossiblePath();
}

void tPoolingLayerBase::reset()
{
    // Nothing needed here...
}

void tPoolingLayerBase::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    out << std::setw(w) << "Pooling Layer:";
    out << std::setw(w) << '-';
    out << std::setw(w) << '-';
    out << std::setw(w) << '-';

    {
        std::ostringstream o;
        o << m_poolRows << "x" << m_poolCols;
        out << std::setw(w) << o.str();
    }

    out << std::setw(w) << 0;

    out << std::setw(w) << 0;

    {
        std::ostringstream o;
        o << (m_inputRows/m_poolRows) << "x" << (m_inputCols/m_poolCols);
        out << std::setw(w) << o.str();
    }

    out << std::endl;
}

std::string tPoolingLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << "pool" << m_poolRows << "x" << m_poolCols;

    return o.str();
}

void tPoolingLayerBase::pack(iWritable* out) const
{
    rho::pack(out, m_inputRows);
    rho::pack(out, m_inputCols);
    rho::pack(out, m_poolRows);
    rho::pack(out, m_poolCols);
}

void tPoolingLayerBase::unpack(iReadable* in)
{
    rho::unpack(in, m_inputRows);
    rho::unpack(in, m_inputCols);
    rho::unpack(in, m_poolRows);
    rho::unpack(in, m_poolCols);
    m_curCount = 0;
    m_maxCount = 0;
}


}   // namespace ml
