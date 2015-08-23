#ifndef __ml_tSplittingLayerBase_h__
#define __ml_tSplittingLayerBase_h__


#include <ml/iLayer.h>


namespace ml
{


class tSplittingLayerBase : public iLayer, public bNonCopyable
{
    public:

        /**
         * Constructs an uninitialized splitting layer. You should call
         * unpack() after using this c'tor.
         */
        tSplittingLayerBase();

        /**
         * Constructs a splitting layer.
         */
        tSplittingLayerBase(u32 numInputDims, u32 numOutputDims);

        /**
         * D'tor.
         */
        ~tSplittingLayerBase();

        /**
         * Adds a layer to receive part of the split input.
         *
         * This method takes ownership of the given layer.
         */
        void addLayer(iLayer* layer, u32 numInputDims, u32 numOutputDims);


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

        struct tLayerRecord
        {
            iLayer* layer;
            u32 numInputDims;
            u32 numOutputDims;
            fml* inputPtr;
            fml* outputErrorPtr;
            tLayerRecord()
            {
                layer = NULL;
                numInputDims = 0;
                numOutputDims = 0;
                inputPtr = NULL;
                outputErrorPtr = NULL;
            }
            tLayerRecord(iLayer* l, u32 i, u32 o)
            {
                layer = l;
                numInputDims = i;
                numOutputDims = o;
                inputPtr = NULL;
                outputErrorPtr = NULL;
            }
        };

        std::vector<tLayerRecord> m_layerRecords;

        u32 m_curCount;
        u32 m_maxCount;
};


}   // namespace ml


#endif   // __ml_tSplittingLayerBase_h__
