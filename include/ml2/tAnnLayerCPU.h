#ifndef __ml2_tAnnLayerCPU_h__
#define __ml2_tAnnLayerCPU_h__


#include <ml2/tAnnLayerBase.h>


namespace ml2
{


class tAnnLayerCPU : public tAnnLayerBase
{
    public:

        /**
         * See tAnnLayerBase::tAnnLayerBase().
         */
        tAnnLayerCPU();

        /**
         * See tAnnLayerBase::tAnnLayerBase().
         */
        tAnnLayerCPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                     u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                     fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tAnnLayerCPU();


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


}   // namespace ml2


#endif   // __ml2_tAnnLayerCPU_h__
