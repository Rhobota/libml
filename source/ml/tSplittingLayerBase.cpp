#include <ml/tSplittingLayerBase.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


tSplittingLayerBase::tSplittingLayerBase()
    : m_numInputDims(0),
      m_numOutputDims(0),
      m_layerRecords(),
      m_curCount(0),
      m_maxCount(0)
{
}

tSplittingLayerBase::tSplittingLayerBase(u32 numInputDims, u32 numOutputDims)
    : m_numInputDims(numInputDims),
      m_numOutputDims(numOutputDims),
      m_layerRecords(),
      m_curCount(0),
      m_maxCount(0)
{
}

tSplittingLayerBase::~tSplittingLayerBase()
{
    if (m_layerRecords.size() > 0)
    {
        throw eLogicError("The subclass should have cleaned this up!");
    }
}

void tSplittingLayerBase::addLayer(iLayer* layer, u32 numInputDims, u32 numOutputDims)
{
    if (!layer)
        throw eInvalidArgument("The layer must not be NULL.");
    if (numInputDims == 0)
        throw eInvalidArgument("numInputDims must be positive.");
    if (numOutputDims == 0)
        throw eInvalidArgument("numOutputDims must be positive.");
    m_layerRecords.push_back(tLayerRecord(layer, numInputDims, numOutputDims));
}

void tSplittingLayerBase::reset()
{
    // Nothing needed here...
}

void tSplittingLayerBase::printLayerInfo(std::ostream& out) const
{
    int w = 25;

    out << std::setw(w) << "Splitting Layer:";
    out << std::setw(w) << '-';
    out << std::setw(w) << '-';
    out << std::setw(w) << '-';

    {
        std::ostringstream o;
        o << m_layerRecords.size() << " sublayers";
        out << std::setw(w) << o.str();
    }

    out << std::setw(w) << '-';
    out << std::setw(w) << '-';

    {
        std::ostringstream o;
        o << m_numOutputDims;
        out << std::setw(w) << o.str();
    }

    out << std::endl;

    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        iLayer* layer = m_layerRecords[i].layer;
        layer->printLayerInfo(out);
    }
}

std::string tSplittingLayerBase::layerInfoString() const
{
    std::ostringstream o;

    o << m_layerRecords.size() << "split";

    return o.str();
}

void tSplittingLayerBase::pack(iWritable* out) const
{
    rho::pack(out, m_numInputDims);
    rho::pack(out, m_numOutputDims);
    rho::pack(out, ((u32)(m_layerRecords.size())));
    for (size_t i = 0; i < m_layerRecords.size(); i++)
    {
        iLayer::writeLayerToStream(m_layerRecords[i].layer, out);
        rho::pack(out, m_layerRecords[i].numInputDims);
        rho::pack(out, m_layerRecords[i].numOutputDims);
    }
}

void tSplittingLayerBase::unpack(iReadable* in)
{
    if (m_layerRecords.size() > 0)
    {
        throw eLogicError("The subclass should have cleaned this up!");
    }

    rho::unpack(in, m_numInputDims);
    rho::unpack(in, m_numOutputDims);

    u32 numLayers;
    rho::unpack(in, numLayers);

    for (u32 i = 0; i < numLayers; i++)
    {
        tLayerRecord rec;
        rec.layer = iLayer::newLayerFromStream(in);
        rho::unpack(in, rec.numInputDims);
        rho::unpack(in, rec.numOutputDims);
        m_layerRecords.push_back(rec);
    }

    m_curCount = 0;
    m_maxCount = 0;
}


}   // namespace ml
