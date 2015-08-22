#ifndef __ml_tReductionLayerBase_h__
#define __ml_tReductionLayerBase_h__


#include <ml/iLayer.h>


namespace ml
{


class tReductionLayerBase : public iLayer, public bNonCopyable
{
    public:

        /**
         * Constructs an uninitialized reduction layer. You should call
         * unpack() after using this c'tor.
         */
        tReductionLayerBase();

        /**
         * Constructs a reduction layer.
         */
        tReductionLayerBase(u32 numInputDims, u32 numOutputDims);

        /**
         * D'tor.
         */
        ~tReductionLayerBase();


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        fml calculateError(const tIO& output, const tIO& target);

        fml calculateError(const std::vector<tIO>& outputs,
                           const std::vector<tIO>& targets);

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

        u32 m_curCount;
        u32 m_maxCount;
};


}   // namespace ml


#endif   // __ml_tReductionLayerBase_h__
