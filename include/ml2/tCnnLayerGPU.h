#ifndef __ml2_tCnnLayerGPU_h__
#define __ml2_tCnnLayerGPU_h__


#include <ml2/tCnnLayerBase.h>


namespace ml2
{


class tCnnLayerGPU : public tCnnLayerBase
{
    public:

        /**
         * See tCnnLayerBase::tCnnLayerBase().
         */
        tCnnLayerGPU();

        /**
         * See tCnnLayerBase::tCnnLayerBase().
         */
        tCnnLayerGPU(nLayerType type, nLayerWeightUpdateRule rule,
                     u32 inputRows, u32 inputCols, u32 inputComponents,
                     u32 kernelRows, u32 kernelCols,
                     u32 kernelStepY, u32 kernelStepX,
                     u32 numKernels,
                     algo::iLCG& lcg, fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tCnnLayerGPU();


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

        void m_initAccum();
        void m_finalize();

        void m_syncWeights_deviceToHost() const;
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


#endif   // __ml2_tCnnLayerGPU_h__
