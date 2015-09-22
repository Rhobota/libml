#ifndef __ml_tDropoutLayerCPU_h__
#define __ml_tDropoutLayerCPU_h__


#include <ml/tDropoutLayerBase.h>


namespace ml
{


class tDropoutLayerCPU : public tDropoutLayerBase
{
    public:

        /**
         * See tDropoutLayerBase::tDropoutLayerBase().
         */
        tDropoutLayerCPU();

        /**
         * See tDropoutLayerBase::tDropoutLayerBase().
         */
        tDropoutLayerCPU(u32 numInputDims, u32 numOutputDims, fml p = FML(0.5));

        /**
         * D'tor.
         */
        ~tDropoutLayerCPU();


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        void takeInput(const fml* input, u32 numInputDims, u32 count);

        const fml* getOutput(u32& numOutputDims, u32& count) const;

        void takeOutputErrorGradients(
                          const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                          const fml* input, u32 numInputDims, u32 inputCount,
                          bool calculateInputErrorGradients);

        const fml* getInputErrorGradients(u32& numInputDims, u32& count) const;

        u32 headerId() const;


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial re-implementation)
        ///////////////////////////////////////////////////////////////////////

        void reset();


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:   (fully re-implemented)
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    private:

        fml* m_output;
        fml* m_inputErrorGradients;

        fml* m_dropMask;
};


}   // namespace ml


#endif   // __ml_tDropoutLayerCPU_h__
