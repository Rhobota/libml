#ifndef __ml_tCnnLayerCPU_h__
#define __ml_tCnnLayerCPU_h__


#include <ml/tCnnLayerBase.h>


namespace ml
{


class tCnnLayerCPU : public tCnnLayerBase
{
    public:

        /**
         * See tCnnLayerBase::tCnnLayerBase().
         */
        tCnnLayerCPU();

        /**
         * See tCnnLayerBase::tCnnLayerBase().
         */
        tCnnLayerCPU(nLayerType type, nLayerWeightUpdateRule rule,
                     u32 inputRows, u32 inputCols, u32 inputComponents,
                     u32 kernelRows, u32 kernelCols,
                     u32 kernelStepY, u32 kernelStepX,
                     u32 numKernels,
                     algo::iLCG& lcg, fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tCnnLayerCPU();


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
        // The iPackable interface:   (partial re-implementation)
        ///////////////////////////////////////////////////////////////////////

        void unpack(iReadable* in);


    private:

        void m_initAccum();
        void m_finalize();


    private:

        fml* m_dw_accum;
        fml* m_db_accum;

        fml* m_A;
        fml* m_a;
        fml* m_dA;
        fml* m_prev_da;

        fml* m_vel;
        fml* m_dw_accum_avg;
};


}   // namespace ml


#endif   // __ml_tCnnLayerCPU_h__
