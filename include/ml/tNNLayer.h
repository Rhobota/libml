#ifndef __ml_tNNLayer_h__
#define __ml_tNNLayer_h__


#include <ml/iLayer.h>

#include <rho/bNonCopyable.h>
#include <rho/algo/tLCG.h>

#include <string>


namespace ml
{


/**
 * Represents a neural network layer. Obviously this call cannot be used
 * directly. You must use one of its subclasses.
 */
class tNNLayer : public iLayer, public bNonCopyable
{
    public:

        /**
         * Possible squashing functions used on the neurons in a neural network layer.
         */
        enum nLayerType {
            kLayerTypeLogistic      = 0, // the logistic function
            kLayerTypeHyperbolic    = 1, // the hyperbolic tangent function
            kLayerTypeSoftmax       = 2, // a softmax group
            kLayerTypeMax           = 3  // marks the max of this enum (do not use)
        };

        static std::string layerTypeToString(nLayerType type);

        static char        layerTypeToChar(nLayerType type);


        /**
         * Possible methods for using the output error gradients to update
         * the weights in a neural network layer.
         */
        enum nLayerWeightUpdateRule {
            kWeightUpRuleNone              = 0,  // no changes will be made to the weights
            kWeightUpRuleFixedLearningRate = 1,  // the standard fixed learning rate method
            kWeightUpRuleMomentum          = 2,  // the momentum learning rate method
            kWeightUpRuleAdaptiveRates     = 3,  // the adaptive learning rates method (for full- or large-batch)
            kWeightUpRuleRPROP             = 4,  // the rprop full-batch method
            kWeightUpRuleRMSPROP           = 5,  // the rmsprop mini-batch method (a mini-batch version of rprop)
            kWeightUpRuleARMS              = 6,  // the adaptive rmsprop method
            kWeightUpRuleMax               = 7   // marks the max of this enum (do not use)
        };

        static std::string weightUpRuleToString(nLayerWeightUpdateRule rule);

        static char        weightUpRuleToChar(nLayerWeightUpdateRule rule);


    private:

};


}   // namespace ml


#endif  // __ml_tNNLayer_h__
