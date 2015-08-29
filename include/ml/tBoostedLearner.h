#ifndef __ml_tBoostedLearner_h__
#define __ml_tBoostedLearner_h__


#include <ml/iLearner.h>


namespace ml
{


class tBoostedLearner : public iLearner, public bNonCopyable
{
    public:

        tBoostedLearner(u32 numInputDims, u32 numOutputDims);

        tBoostedLearner(iReadable* in);

        ~tBoostedLearner();

        void addLearner(iLearner* learner, fml weight);  // <-- takes ownership of the learner

        void setOutputPerformanceEvaluator(iOutputPerformanceEvaluator* evaluator); // <-- takes ownership


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

        iOutputPerformanceEvaluator* getOutputPerformanceEvaluator();

        void reset();

        void printLearnerInfo(std::ostream& out) const;

        std::string learnerInfoString() const;

        u32 headerId() const;


        ///////////////////////////////////////////////////////////////////////
        // The iPackable interface:
        ///////////////////////////////////////////////////////////////////////

        void pack(iWritable* out) const;
        void unpack(iReadable* in);


    private:

        u32 m_numInputDims;
        u32 m_numOutputDims;

        std::vector< std::pair<iLearner*, fml> > m_learners;

        iOutputPerformanceEvaluator* m_evaluator;
};


}   // namespace ml


#endif   // __ml_tBoostedLearner_h__
