#ifndef __ml_tLayeredLearnerCPU_h__
#define __ml_tLayeredLearnerCPU_h__


#include <ml/tLayeredLearnerBase.h>


namespace ml
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


    protected:

        void m_calculate_output_da(const fml* output, fml* target, u32 dims, u32 count);


    private:

};


}   // namespace ml


#endif   // __ml_tLayeredLearnerCPU_h__
