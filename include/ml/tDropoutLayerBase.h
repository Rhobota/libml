#ifndef __ml_tDropoutLayerBase_h__
#define __ml_tDropoutLayerBase_h__


#include <ml/iLayer.h>


namespace ml
{


class tDropoutLayerBase : public iLayer, public bNonCopyable
{
    public:

        /**
         * Constructs an uninitialized dropout layer. You should call
         * unpack() after using this c'tor.
         */
        tDropoutLayerBase();

        /**
         * Constructs a dropout layer with the given p value.
         */
        tDropoutLayerBase(u32 numInputDims, u32 numOutputDims, fml p = FML(0.5));

        /**
         * D'tor.
         */
        ~tDropoutLayerBase();


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

        fml m_p;

        u32 m_curCount;
        u32 m_maxCount;
};


}   // namespace ml


#endif   // __ml_tDropoutLayerBase_h__
