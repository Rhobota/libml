#ifndef __ml2_tCnnLayerBase_h__
#define __ml2_tCnnLayerBase_h__


#include <ml2/tNNLayer.h>


namespace ml2
{


class tCnnLayerBase : public tNNLayer
{
    public:

        /**
         * Constructs an empty Cnn. You probably want to call unpack if you use
         * this c'tor.
         */
        tCnnLayerBase();

        /**
         * Constructs this layer to use the specified layer type, weight
         * update rule, input image size, and convolution kernel size.
         *
         * See tAnnLayerBase::tAnnLayerBase() for a description of 'type'
         * and 'rule'.
         *
         * A subsequent call to this object's takeInput() method should have
         * 'numInputDims' set to 'inputRows*inputCols*inputComponents'.
         *
         * The output of this layer will have dimensionality of
         * 'inputRows*inputCols*numKernels' if 'kernelStepY' and
         * 'kernelStepX' are both 1.
         *
         * Both 'kernelRows' and 'kernelCols' must be an odd number.
         * Consider 3, 5, or 7 for each. The larger the kernel size,
         * the less efficient the computation will be.
         */
        tCnnLayerBase(nLayerType type, nLayerWeightUpdateRule rule,
                      u32 inputRows, u32 inputCols, u32 inputComponents,
                      u32 kernelRows, u32 kernelCols,
                      u32 kernelStepY, u32 kernelStepX,
                      u32 numKernels,
                      algo::iLCG& lcg, fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tCnnLayerBase();

        /**
         * See tAnnLayerBase::setAlpha() for a description of this method.
         */
        void setAlpha(fml alpha);

        /**
         * See tAnnLayerBase::setViscosity() for a description of this method.
         */
        void setViscosity(fml viscosity);


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        fml calculateError(const tIO& output, const tIO& target);

        fml calculateError(const std::vector<tIO>& outputs,
                           const std::vector<tIO>& targets);

        void reset();

        void printLayerInfo(std::ostream& out) const;

        std::string layerInfoString() const;


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    protected:

        void m_initWeights(algo::iLCG& lcg,
                           fml randWeightMin,
                           fml randWeightMax);


    protected:

        nLayerType             m_type;
        nLayerWeightUpdateRule m_rule;

        fml m_alpha;
        fml m_viscosity;

        u32  m_inputRows;
        u32  m_inputCols;
        u32  m_inputComponents;

        u32  m_kernelRows;
        u32  m_kernelCols;
        u32  m_kernelStepY;
        u32  m_kernelStepX;
        u32  m_numKernels;

        u32  m_curCount;
        u32  m_maxCount;

        fml* m_w;
        fml* m_b;

        fml* m_w_orig;
        fml* m_b_orig;
};


}   // namespace ml2


#endif   // __ml2_tCnnLayerBase_h__
