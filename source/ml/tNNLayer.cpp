#include <ml2/tNNLayer.h>

#include <cassert>


namespace ml2
{


std::string tNNLayer::layerTypeToString(nLayerType type)
{
    switch (type)
    {
        case kLayerTypeLogistic:
            return "logistic";
        case kLayerTypeHyperbolic:
            return "hyperbolic";
        case kLayerTypeSoftmax:
            return "softmax";
        default:
            assert(false);
    }
}


char tNNLayer::layerTypeToChar(nLayerType type)
{
    switch (type)
    {
        case kLayerTypeLogistic:
            return 'l';
        case kLayerTypeHyperbolic:
            return 'h';
        case kLayerTypeSoftmax:
            return 's';
        default:
            assert(false);
    }
}


std::string tNNLayer::weightUpRuleToString(nLayerWeightUpdateRule rule)
{
    switch (rule)
    {
        case kWeightUpRuleNone:
            return "none";
        case kWeightUpRuleFixedLearningRate:
            return "fixed rate";
        case kWeightUpRuleMomentum:
            return "momentum";
        case kWeightUpRuleAdaptiveRates:
            return "adaptive rates";
        case kWeightUpRuleRPROP:
            return "rprop";
        case kWeightUpRuleRMSPROP:
            return "rmsprop";
        case kWeightUpRuleARMS:
            return "arms";
        default:
            assert(false);
    }
}


char tNNLayer::weightUpRuleToChar(nLayerWeightUpdateRule rule)
{
    switch (rule)
    {
        case kWeightUpRuleNone:
            return 'n';
        case kWeightUpRuleFixedLearningRate:
            return 'f';
        case kWeightUpRuleMomentum:
            return 'm';
        case kWeightUpRuleAdaptiveRates:
            return 'a';
        case kWeightUpRuleRPROP:
            return 'r';
        case kWeightUpRuleRMSPROP:
            return 'R';
        case kWeightUpRuleARMS:
            return 'A';
        default:
            assert(false);
    }
}


}  // namespace ml2
