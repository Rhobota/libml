#ifndef __ml_tScalingLayerGPU_h__
#define __ml_tScalingLayerGPU_h__


#include <ml/tScalingLayerBase.h>


namespace ml
{


class tScalingLayerGPU : public tScalingLayerBase
{
    public:

        /**
         * See tScalingLayerBase::tScalingLayerBase().
         */
        tScalingLayerGPU();

        /**
         * See tScalingLayerBase::tScalingLayerBase().
         */
        tScalingLayerGPU(u32 numInputDims, u32 numOutputDims, fml scaleFactor);

        /**
         * D'tor.
         */
        ~tScalingLayerGPU();


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

        fml* m_gpu_output;
        fml* m_gpu_inputErrorGradients;
};


}   // namespace ml


#endif   // __ml_tScalingLayerGPU_h__
