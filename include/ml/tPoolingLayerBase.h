#ifndef __ml_tPoolingLayerBase_h__
#define __ml_tPoolingLayerBase_h__


#include <ml/iLayer.h>


namespace ml
{


class tPoolingLayerBase : public iLayer, public bNonCopyable
{
    public:

        /**
         * Constructs an uninitialized pooling layer. You should call
         * unpack() after using this c'tor.
         */
        tPoolingLayerBase();

        /**
         * Constructs a pooling layer. Yay! Right now only MAX-pooling is
         * implemented. I'd recommend doing 2x2 pooling, because that already
         * cuts down the dimensionality by a factor of 4, which is quite a lot
         * really. You probably don't need any more reduction than that from
         * one layer, especially a simple layer like this one.
         */
        tPoolingLayerBase(u32 inputRows, u32 inputCols, u32 inputComponents,
                          u32 poolRows = 2, u32 poolCols = 2);

        /**
         * D'tor.
         */
        ~tPoolingLayerBase();


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

        u32 m_inputRows;
        u32 m_inputCols;
        u32 m_inputComponents;

        u32 m_poolRows;
        u32 m_poolCols;

        u32 m_curCount;
        u32 m_maxCount;
};


}   // namespace ml


#endif   // __ml_tPoolingLayerBase_h__
