#ifndef __ml_tScalingLayerCPU_h__
#define __ml_tScalingLayerCPU_h__


#include <ml/tScalingLayerBase.h>


namespace ml
{


class tScalingLayerCPU : public tScalingLayerBase
{
    public:

        /**
         * See tScalingLayerBase::tScalingLayerBase().
         */
        tScalingLayerCPU();

        /**
         * See tScalingLayerBase::tScalingLayerBase().
         */
        tScalingLayerCPU(u32 numInputDims, u32 numOutputDims, fml scaleFactor);

        /**
         * D'tor.
         */
        ~tScalingLayerCPU();


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


    private:

        fml* m_output;
        fml* m_inputErrorGradients;
};


}   // namespace ml


#endif   // __ml_tScalingLayerCPU_h__
