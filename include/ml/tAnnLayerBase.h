#ifndef __ml_tAnnLayerBase_h__
#define __ml_tAnnLayerBase_h__


#include <ml/tNNLayer.h>


namespace ml
{


class tAnnLayerBase : public tNNLayer
{
    public:

        /**
         * Constructs an empty ann. You probably want to call unpack if you use
         * this c'tor.
         */
        tAnnLayerBase();

        /**
         * Constructs this layer to use the specified layer type and
         * weight update rule.
         *
         * You must also setup the learning parameters associated with
         * the weight update rule for this layer. Below describes the
         * parameters needed for each rule:
         *
         *    - kWeightUpRuleNone
         *         -- no extra parameters needed
         *
         *    - kWeightUpRuleFixedLearningRate
         *         -- requires setAlpha()
         *
         *    - kWeightUpRuleMomentum
         *         -- requires setAlpha() and setViscosity()
         *
         *    - kWeightUpRuleAdaptiveRates
         *         -- requires setAlpha()
         *         -- requires using full- or large-batch learning
         *
         *    - kWeightUpRuleRPROP
         *         -- no extra parameters needed
         *         -- requires full-batch learning
         *
         *    - kWeightUpRuleRMSPROP
         *         -- requires setAlpha()
         *         -- this is a mini-batch version of the rprop method
         *
         *    - kWeightUpRuleARMS
         *         -- requires setAlpha()
         *         -- very similar to RMSPROP, but has an adaptive alpha
         *
         * A subsequent call to this object's takeInput() method should
         * contain one or more images, where each image:
         *   - is stored in row-major format,
         *   - is 'inputCols'x'inputRows' in size, and
         *   - has 'inputComponents' channels.
         * Therefore, a subsequent call to this object's takeInput() method
         * should have 'numInputDims' set to 'inputRows*inputCols*inputComponents',
         * and should have 'count' set to the number of sequential images within
         * the input vector.
         *
         * Note: The input doesn't have to actually be an image. You can input
         * arbitrary data here, in which case you can set 'inputRows' equal to the
         * dimensionality of the input, and set 'inputCols' and 'inputComponents'
         * both equal to 1.
         *
         * The output of this layer will have dimensionality of 'numNeurons'.
         */
        tAnnLayerBase(nLayerType type, nLayerWeightUpdateRule rule,
                      u32 inputRows, u32 inputCols, u32 inputComponents,
                      u32 numNeurons,
                      algo::iLCG& lcg, fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * D'tor.
         */
        ~tAnnLayerBase();

        /**
         * Resets the layer type. See tAnnLayerBase::tAnnLayerBase() for
         * details.
         */
        void setLayerType(nLayerType type)                           { m_type = type; }

        /**
         * Resets the layer weight update rule. See tAnnLayerBase::tAnnLayerBase()
         * for details.
         */
        void setLayerWeightUpdateRule(nLayerWeightUpdateRule rule)   { m_rule = rule; }

        /**
         * Sets the alpha parameter for this layer. The alpha parameter is
         * the "fixed learning rate" parameter, used when the weight update
         * rule is kWeightUpRuleFixedLearningRate.
         *
         * This parameter is also used when the weight update rule
         * is kWeightUpRuleMomentum.
         *
         * This parameter is also used when the weight update rule
         * is kWeightUpRuleAdaptiveRates for the "base rate".
         * Note: If you use kWeightUpRuleAdaptiveRates, you must
         * use full- or large-batch learning.
         *
         * This parameter is also used when the weight update rule
         * is kWeightUpRuleRMSPROP or kWeightUpRuleARMS.
         */
        void setAlpha(fml alpha);

        /**
         * Sets the viscosity of the network's weight velocities when using
         * the momentum weight update rule (kWeightUpRuleMomentum).
         */
        void setViscosity(fml viscosity);


        ///////////////////////////////////////////////////////////////////////
        // Misc getters:
        ///////////////////////////////////////////////////////////////////////

        nLayerType              layerType()              const   { return m_type; }
        nLayerWeightUpdateRule  layerWeightUpdateRule()  const   { return m_rule; }

        fml alpha()      const    { return m_alpha; }
        fml viscosity()  const    { return m_viscosity; }

        u32 inputRows()       const    { return m_inputRows; }
        u32 inputCols()       const    { return m_inputCols; }
        u32 inputComponents() const    { return m_inputComponents; }
        u32 numNeurons()      const    { return m_numNeurons; }

        virtual void currentState(std::vector<tIO>& weights, tIO& biases, tIO& outputs) const = 0;


        ///////////////////////////////////////////////////////////////////////
        // The iLayer interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        void reset();

        void printLayerInfo(std::ostream& out) const;

        std::string layerInfoString() const;


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    protected:

        void m_validate();

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
        u32  m_numInputDims;
        u32  m_numNeurons;

        u32  m_curCount;
        u32  m_maxCount;

        fml* m_w;
        fml* m_b;

        fml* m_w_orig;
        fml* m_b_orig;
};


}   // namespace ml


#endif   // __ml_tAnnLayerBase_h__
