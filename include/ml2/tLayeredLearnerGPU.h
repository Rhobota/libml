#ifndef __ml2_tLayeredLearnerGPU_h__
#define __ml2_tLayeredLearnerGPU_h__


#include <ml2/tLayeredLearnerBase.h>


namespace ml2
{


class tLayeredLearnerGPU : public tLayeredLearnerBase
{
    public:

        tLayeredLearnerGPU(u32 numInputDims, u32 numOutputDims);

        tLayeredLearnerGPU(iReadable* in);

        ~tLayeredLearnerGPU();


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

        void m_copyTo_gpu_buf_1(const fml* local_buf, u32 size);
        void m_copyTo_gpu_buf_2(const fml* local_buf, u32 size);

        void m_copyTo_local_buf_1(const fml* gpu_buf, u32 size);


    private:

        fml* m_gpu_buf_1;
        u32  m_gpu_buf_1_size;

        fml* m_gpu_buf_2;
        u32  m_gpu_buf_2_size;

        fml* m_local_buf_1;
        u32  m_local_buf_1_size;
};


}   // namespace ml2


#endif   // __ml2_tLayeredLearnerGPU_h__
