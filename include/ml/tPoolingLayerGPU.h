#ifndef __ml_tPoolingLayerGPU_h__
#define __ml_tPoolingLayerGPU_h__


#include <ml/tPoolingLayerBase.h>


namespace ml
{


class tPoolingLayerGPU : public tPoolingLayerBase
{
    public:

        /**
         * See tPoolingLayerBase::tPoolingLayerBase().
         */
        tPoolingLayerGPU();

        /**
         * See tPoolingLayerBase::tPoolingLayerBase().
         */
        tPoolingLayerGPU(u32 inputRows, u32 inputCols, u32 inputComponents);

        /**
         * See tPoolingLayerBase::tPoolingLayerBase().
         */
        tPoolingLayerGPU(u32 inputRows, u32 inputCols, u32 inputComponents,
                         u32 poolRows, u32 poolCols);

        /**
         * D'tor.
         */
        ~tPoolingLayerGPU();


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

        fml* m_gpu_a;
        fml* m_gpu_prev_da;
};


}   // namespace ml


#endif   // __ml_tPoolingLayerGPU_h__
