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
        tAnnLayerCPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                     u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                     fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * See tAnnLayerBase::tAnnLayerBase().
         */
        tAnnLayerCPU(iReadable* in);

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


    private:

};


}   // namespace ml2


#endif   // __ml2_tAnnLayerCPU_h__
