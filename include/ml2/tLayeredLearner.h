#ifndef __ml2_tLayeredLearner_h__
#define __ml2_tLayeredLearner_h__


#include <ml2/iLearner.h>
#include <ml2/iLayer.h>


namespace ml2
{


class tLayeredLearner : public iLearner, public bNonCopyable
{
    public:

        tLayeredLearner(u32 numInputDims, u32 numOutputDims);

        ~tLayeredLearner();

        void addLayer(iLayer* layer);

        ///////////////////////////////////////////////////////////////////////
        // iLearner interface:
        ///////////////////////////////////////////////////////////////////////

        void addExample(const tIO& input, const tIO& target);

        void update();

        void evaluate(const tIO& input, tIO& output);

        void evaluateBatch(const std::vector<tIO>& inputs,
                                 std::vector<tIO>& outputs);

        void evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                           std::vector<tIO>::const_iterator inputEnd,
                           std::vector<tIO>::iterator outputStart);

        fml calculateError(const tIO& output, const tIO& target);

        fml calculateError(const std::vector<tIO>& outputs,
                           const std::vector<tIO>& targets);

        void reset();

        void printLearnerInfo(std::ostream& out) const;

        std::string learnerInfoString() const;

    private:

        void m_growInputMatrix(u32 newSize);
        void m_growTargetMatrix(u32 newSize);

        void m_copyToInputMatrix(const tIO& input);
        void m_copyToTargetMatrix(const tIO& target);

    private:

        std::vector<iLayer*> m_layers;

        u32 m_numInputDims;
        u32 m_numOutputDims;

        fml* m_inputMatrix;
        u32  m_inputMatrixSize;
        u32  m_inputMatrixUsed;

        fml* m_targetMatrix;
        u32  m_targetMatrixSize;
        u32  m_targetMatrixUsed;
};


}   // namespace ml2


#endif   // __ml2_tLayeredLearner_h__
