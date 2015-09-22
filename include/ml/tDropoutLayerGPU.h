#ifndef __ml_tDropoutLayerGPU_h__
#define __ml_tDropoutLayerGPU_h__


#include <ml/tDropoutLayerBase.h>


namespace ml
{


class tDropoutLayerGPU : public tDropoutLayerBase
{
    public:

        /**
         * See tDropoutLayerBase::tDropoutLayerBase().
         */
        tDropoutLayerGPU();

        /**
         * See tDropoutLayerBase::tDropoutLayerBase().
         */
        tDropoutLayerGPU(u32 numInputDims, u32 numOutputDims, u32 rndSeed, fml p = FML(0.5));

        /**
         * D'tor.
         */
        ~tDropoutLayerGPU();


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        void takeInput(const fml* input, u32 numInputDims, u32 count,
                       bool isTrainMode, iLayer* prevLayer);

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

        fml* m_gpu_output;
        fml* m_gpu_inputErrorGradients;
};


}   // namespace ml


#endif   // __ml_tDropoutLayerGPU_h__
