#ifndef __ml_tWrappedGPULayer_h__
#define __ml_tWrappedGPULayer_h__


#include <ml/iLayer.h>


namespace ml
{


/**
 * This class is useful for testing GPU layers. It takes host
 * memory as input and outputs host memory, so testing is easy.
 * It will copy to/from host memory and GPU memory internally,
 * so that the wrapped GPU layer receives GPU memory as expected,
 * and can output GPU memory as expected.
 */
class tWrappedGPULayer : public iLayer, public bNonCopyable
{
    public:

        /**
         * Constructs for the given wrapped GPU layer.
         * The wrapped layer passed here MUST be a GPU layer.
         *
         * This class takes ownership of the wrapped layer.
         * (I.e., this object will call delete on it for you.)
         */
        tWrappedGPULayer(u32 numInputDims, u32 numOutputDims, iLayer* wrappedLayer);

        /**
         * D'tor.
         */
        ~tWrappedGPULayer();


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:
        ///////////////////////////////////////////////////////////////////////

        void reset();

        void printLayerInfo(std::ostream& out) const;

        std::string layerInfoString() const;

        void takeInput(const fml* input, u32 numInputDims, u32 count);

        const fml* getOutput(u32& numOutputDims, u32& count) const;

        void takeOutputErrorGradients(
                          const fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                          const fml* input, u32 numInputDims, u32 inputCount,
                          bool calculateInputErrorGradients);

        const fml* getInputErrorGradients(u32& numInputDims, u32& count) const;

        u32 headerId() const;


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    private:

        u32 m_numInputDims;
        u32 m_numOutputDims;

        iLayer* m_wrappedLayer;

        u32 m_curCount;
        u32 m_maxCount;

        fml* m_gpu_input;
        fml* m_output;

        fml* m_gpu_outputErrorGradients;
        fml* m_inputErrorGradients;
};


}   // namespace ml


#endif   // __ml_tWrappedGPULayer_h__
