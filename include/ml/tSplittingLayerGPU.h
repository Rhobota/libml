#ifndef __ml_tSplittingLayerGPU_h__
#define __ml_tSplittingLayerGPU_h__


#include <ml/tSplittingLayerBase.h>


namespace ml
{


class tSplittingLayerGPU : public tSplittingLayerBase
{
    public:

        /**
         * See tSplittingLayerBase::tSplittingLayerBase().
         */
        tSplittingLayerGPU();

        /**
         * See tSplittingLayerBase::tSplittingLayerBase().
         */
        tSplittingLayerGPU(u32 numInputDims, u32 numOutputDims);

        /**
         * D'tor.
         */
        ~tSplittingLayerGPU();


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


    private:

        fml* m_gpu_a;
        fml* m_gpu_prev_da;
};


}   // namespace ml


#endif   // __ml_tSplittingLayerGPU_h__
