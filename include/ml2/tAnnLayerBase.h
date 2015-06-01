#ifndef __ml2_tAnnLayerBase_h__
#define __ml2_tAnnLayerBase_h__


#include <ml2/iLayer.h>

#include <rho/bNonCopyable.h>
#include <rho/algo/tLCG.h>


namespace ml2
{


class tAnnLayerBase : public iLayer, public bNonCopyable
{
    public:

        /**
         * Possible squashing functions used on the neurons in this layer.
         */
        enum nAnnLayerType {
            kLayerTypeLogistic      = 0, // the logistic function
            kLayerTypeHyperbolic    = 1, // the hyperbolic tangent function
            kLayerTypeSoftmax       = 2, // a softmax group
            kLayerTypeMax           = 3  // marks the max of this enum (do not use)
        };

        /**
         * Possible methods for using the output error gradients to update
         * the weights in this layer.
         */
        enum nAnnLayerWeightUpdateRule {
            kWeightUpRuleNone              = 0,  // no changes will be made to the weights
            kWeightUpRuleFixedLearningRate = 1,  // the standard fixed learning rate method
            kWeightUpRuleMomentum          = 2,  // the momentum learning rate method
            kWeightUpRuleAdaptiveRates     = 3,  // the adaptive learning rates method (for full- or large-batch)
            kWeightUpRuleRPROP             = 4,  // the rprop full-batch method
            kWeightUpRuleRMSPROP           = 5,  // the rmsprop mini-batch method (a mini-batch version of rprop)
            kWeightUpRuleARMS              = 6,  // the adaptive rmsprop method
            kWeightUpRuleMax               = 7   // marks the max of this enum (do not use)
        };

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
         */
        tAnnLayerBase(nAnnLayerType type, nAnnLayerWeightUpdateRule rule,
                      u32 numInputDims, u32 numNeurons, algo::iLCG& lcg,
                      fml randWeightMin = -1.0, fml randWeightMax = 1.0);

        /**
         * Constructs a tAnnLayerBase from a stream.
         */
        tAnnLayerBase(iReadable* in);

        /**
         * D'tor.
         */
        ~tAnnLayerBase();

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

        nAnnLayerType             m_type;
        nAnnLayerWeightUpdateRule m_rule;

        fml m_alpha;
        fml m_viscosity;

        u32  m_numInputDims;
        u32  m_numNeurons;

        u32  m_curCount;
        u32  m_maxCount;

        fml* m_w;
        fml* m_b;

        fml* m_w_orig;
        fml* m_b_orig;
};


}   // namespace ml2


#endif   // __ml2_tAnnLayerBase_h__
