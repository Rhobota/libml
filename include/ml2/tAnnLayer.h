#ifndef __ml2_tAnnLayer_h__
#define __ml2_tAnnLayer_h__


#include <ml2/iLayer.h>


namespace ml2
{


class tAnnLayer : public iLayer
{
    public:

        void takeInput(fml* input, u32 numInputDims, u32 count);

        fml* getOutput(u32& numOutputDims, u32& count) const;

        void takeOutputErrorGradients(
                          fml* outputErrorGradients, u32 numOutputDims, u32 outputCount,
                          fml* input, u32 numInputDims, u32 inputCount);

        fml* getInputErrorGradients(u32& numInputDims, u32& count) const;

    private:


};


}   // namespace ml2


#endif   // __ml2_tAnnLayer_h__
