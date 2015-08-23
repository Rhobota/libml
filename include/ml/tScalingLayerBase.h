#ifndef __ml_tScalingLayerBase_h__
#define __ml_tScalingLayerBase_h__


#include <ml/iLayer.h>


namespace ml
{


class tScalingLayerBase : public iLayer, public bNonCopyable
{
    public:

        /**
         * Constructs an uninitialized scaling layer. You should call
         * unpack() after using this c'tor.
         */
        tScalingLayerBase();

        /**
         * Constructs a scaling layer with the given scale factor.
         */
        tScalingLayerBase(u32 numInputDims, u32 numOutputDims, fml scaleFactor);

        /**
         * D'tor.
         */
        ~tScalingLayerBase();


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        void reset();

        void printLayerInfo(std::ostream& out) const;

        std::string layerInfoString() const;


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    protected:

        u32 m_numInputDims;
        u32 m_numOutputDims;

        fml m_scaleFactor;

        u32 m_curCount;
        u32 m_maxCount;
};


}   // namespace ml


#endif   // __ml_tScalingLayerBase_h__
