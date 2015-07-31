#ifndef __ml_tPoolingLayerCPU_h__
#define __ml_tPoolingLayerCPU_h__


#include <ml/tPoolingLayerBase.h>


namespace ml
{


class tPoolingLayerCPU : public tPoolingLayerBase
{
    public:

        /**
         * See tPoolingLayerBase::tPoolingLayerBase().
         */
        tPoolingLayerCPU();

        /**
         * D'tor.
         */
        ~tPoolingLayerCPU();


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

        fml* m_a;
        fml* m_prev_da;
};


}   // namespace ml


#endif   // __ml_tPoolingLayerCPU_h__
