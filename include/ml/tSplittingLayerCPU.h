#ifndef __ml_tSplittingLayerCPU_h__
#define __ml_tSplittingLayerCPU_h__


#include <ml/tSplittingLayerBase.h>


namespace ml
{


class tSplittingLayerCPU : public tSplittingLayerBase
{
    public:

        /**
         * See tSplittingLayerBase::tSplittingLayerBase().
         */
        tSplittingLayerCPU();

        /**
         * See tSplittingLayerBase::tSplittingLayerBase().
         */
        tSplittingLayerCPU(u32 numInputDims, u32 numOutputDims);

        /**
         * D'tor.
         */
        ~tSplittingLayerCPU();


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

        fml* m_a;
        fml* m_prev_da;
};


}   // namespace ml


#endif   // __ml_tSplittingLayerCPU_h__
