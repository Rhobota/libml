#ifndef __ml2_tCnnLayerCPU_h__
#define __ml2_tCnnLayerCPU_h__


#include <ml2/tAnnLayerCPU.h>


namespace ml2
{


class tCnnLayerCPU : public tAnnLayerCPU
{
    public:

        /**
         * See tAnnLayerCPU::tAnnLayerCPU().
         */
        tCnnLayerCPU();

        /**
         * TODO -- Write this comment.
         */
        tCnnLayerCPU(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                     u32 numInputRows, u32 numInputCols, u32 numInputComponents,
                     u32 numFeatureMapRows, u32 numFeatureMapCols, u32 numFeatureMaps,
                     algo::iLCG& lcg, fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tCnnLayerCPU();


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial re-implementation)
        ///////////////////////////////////////////////////////////////////////

        void takeInput(const fml* input, u32 numInputDims, u32 count);

        const fml* getOutput(u32& numOutputDims, u32& count) const;

        void takeOutputErrorGradients(
                          const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                          const fml* input, u32 numInputDims, u32 inputCount,
                          bool calculateInputErrorGradients);

        const fml* getInputErrorGradients(u32& numInputDims, u32& count) const;

        u32 headerId() const;

        void reset();


    private:

};


}   // namespace ml2


#endif   // __ml2_tCnnLayerCPU_h__
