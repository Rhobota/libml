#ifndef __ml_tReductionLayerCPU_h__
#define __ml_tReductionLayerCPU_h__


#include <ml/tReductionLayerBase.h>


namespace ml
{


class tReductionLayerCPU : public tReductionLayerBase
{
    public:

        /**
         * See tReductionLayerBase::tReductionLayerBase().
         */
        tReductionLayerCPU();

        /**
         * See tReductionLayerBase::tReductionLayerBase().
         */
        tReductionLayerCPU(u32 numInputDims, u32 numOutputDims);

        /**
         * D'tor.
         */
        ~tReductionLayerCPU();


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

        fml* m_output;
        fml* m_inputErrorGradients;
};


}   // namespace ml


#endif   // __ml_tReductionLayerCPU_h__
