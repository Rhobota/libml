#ifndef __ml2_tAnnLayerGPU_h__
#define __ml2_tAnnLayerGPU_h__


#include <ml2/tAnnLayerBase.h>


namespace ml2
{


class tAnnLayerGPU : public tAnnLayerBase
{
    public:

        /**
         * See tAnnLayerBase::tAnnLayerBase().
         */
        tAnnLayerGPU();

        /**
         * See tAnnLayerBase::tAnnLayerBase().
         */
        tAnnLayerGPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                     u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                     fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tAnnLayerGPU();


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

        void m_syncWeights_deviceToHost();
        void m_syncWeights_hostToDevice();


    private:

        void* m_cublasContext;

        fml* m_gpu_w;
        fml* m_gpu_b;

        fml* m_gpu_dw_accum;
        fml* m_gpu_db_accum;

        fml* m_gpu_A;
        fml* m_gpu_a;
        fml* m_gpu_dA;
        fml* m_gpu_prev_da;

        fml* m_gpu_vel;
        fml* m_gpu_dw_accum_avg;

        fml* m_gpu_uniqueKeys;
        fml* m_gpu_columnSums;
        fml* m_gpu_ones_vector;
};


}   // namespace ml2


#endif   // __ml2_tAnnLayerGPU_h__
