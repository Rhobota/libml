#include <ml/tPoolingLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


tPoolingLayerBase::tPoolingLayerBase()
    : m_inputRows(0),
      m_inputCols(0),
      m_inputComponents(0),
      m_poolRows(0),
      m_poolCols(0),
      m_curCount(0),
      m_maxCount(0)
{
}

tPoolingLayerBase::tPoolingLayerBase(u32 inputRows, u32 inputCols, u32 inputComponents,
                                     u32 poolRows, u32 poolCols)
    : m_inputRows(inputRows),
      m_inputCols(inputCols),
      m_inputComponents(inputComponents),
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

fml tPoolingLayerBase::calculateError(const tIO& output, const tIO    & target)
{
        throw eImpossiblePath();
}

fml tPoolingLayerBase::calculateError    (const std::vector<tIO>& outputs,
                                      const std::vector<tIO>& targets)
{
    throw eImpossiblePath();
    }

void tPoolingLayerBase    ::reset()
{
        // Nothing needed here    ...
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

    out << std::setw(w) << '-';
    out << std::setw(w) << '-';

    {
        std::ostringstream o;
        o << (m_inputRows/m_poolRows) << "x" << (m_inputCols/m_poolCols) << "x" << m_inputComponents
          << " = " << ((m_inputRows/m_poolRows)*(m_inputCols/m_poolCols)*m_inputComponents);
        out << std::setw(w) << o.str();
    }

    out << std::endl;
}

std::string tPoolingLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << m_poolRows << "x" << m_poolCols << "p";

    return o.str();
}

void tPoolingLayerBase::pack(iWritable* out) const
{
    rho::pack(out, m_inputRows);
    rho::pack(out, m_inputCols);
    rho::pack(out, m_inputComponents);
    rho::pack(out, m_poolRows);
    rho::pack(out, m_poolCols);
}

void tPoolingLayerBase::unpack(iReadable* in)
{
    rho::unpack(in, m_inputRows);
    rho::unpack(in, m_inputCols);
    rho::unpack(in, m_inputComponents);
    rho::unpack(in, m_poolRows);
    rho::unpack(in, m_poolCols);
    m_curCount = 0;
    m_maxCount = 0;
}


}   // namespace ml
