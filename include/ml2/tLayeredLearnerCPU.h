#ifndef __ml2_tLayeredLearnerCPU_h__
#define __ml2_tLayeredLearnerCPU_h__


#include <ml2/tLayeredLearnerBase.h>


namespace ml2
{


class tLayeredLearnerCPU : public tLayeredLearnerBase
{
    public:

        tLayeredLearnerCPU(u32 numInputDims, u32 numOutputDims);

        tLayeredLearnerCPU(iReadable* in);

        ~tLayeredLearnerCPU();


        ///////////////////////////////////////////////////////////////////////
        // iLearner interface:   (partial interface only)
        ///////////////////////////////////////////////////////////////////////

        void update();

        void evaluate(const tIO& input, tIO& output);

        void evaluateBatch(const std::vector<tIO>& inputs,
                                 std::vector<tIO>& outputs);

        void evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                           std::vector<tIO>::const_iterator inputEnd,
                           std::vector<tIO>::iterator outputStart);

        u32 headerId() const;


    private:

};


}   // namespace ml2


#endif   // __ml2_tLayeredLearnerCPU_h__
