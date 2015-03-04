#ifndef __ml2_iLearner_h__
#define __ml2_iLearner_h__


#include <ml2/common.h>

#include <rho/iPackable.h>


namespace ml2
{


/**
 * This learner learns with vector data input and vector data output.
 */
class iLearner : public iPackable
{
    public:

        /**
         * Shows the learner one example. The learner will calculate error rates
         * (or whatever it does with examples), then accumulate the error in some
         * way or another.
         *
         * The learner will not become smarter by calling this method. You must
         * subsequently call update().
         */
        virtual void addExample(const tIO& input, const tIO& target) = 0;

        /**
         * Updates the learner to account for all the examples it's seen since
         * the last call to update().
         *
         * The accumulated error rates (or whatever) are then cleared.
         */
        virtual void update() = 0;

        /**
         * Uses the current knowledge of the learner to evaluate the given input.
         */
        virtual void evaluate(const tIO& input, tIO& output) = 0;

        /**
         * Uses the current knowledge of the learner to evaluate the given inputs.
         *
         * This is the same as the above version of evaluate(), but this one
         * does a batch-evaluate, which is more efficient for most learners
         * to perform.
         */
        virtual void evaluateBatch(const std::vector<tIO>& inputs,
                                         std::vector<tIO>& outputs) = 0;

        /**
         * Uses the current knowledge of the learner to evaluate the given inputs.
         *
         * This is the same as the above version of evaluateBatch(), but this one
         * takes iterators so that you can avoid copying data in some cases.
         */
        virtual void evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                   std::vector<tIO>::const_iterator inputEnd,
                                   std::vector<tIO>::iterator outputStart) = 0;

        /**
         * Asks the learner to calculate the error between the given output
         * and the given target. For example, the learner may calculate
         * the standard squared error or the cross-entropy loss, if one of
         * those is appropriate. Or the learner may do something else.
         */
        virtual fml calculateError(const tIO& output, const tIO& target) = 0;

        /**
         * Asks the learner to calculate the error between all the given
         * output/target pairs. For example, the learner may calculate
         * the average standard squared error or the average cross-entropy
         * loss, if one of those is appropriate. Or the learner may do
         * something else.
         */
        virtual fml calculateError(const std::vector<tIO>& outputs,
                                   const std::vector<tIO>& targets) = 0;

        /**
         * Resets the learner to its initial state.
         */
        virtual void reset() = 0;

        /**
         * Prints the learner's configuration in a readable format.
         */
        virtual void printLearnerInfo(std::ostream& out) const = 0;

        /**
         * Returns a single-line version of printLearnerInfo().
         */
        virtual std::string learnerInfoString() const = 0;

        /**
         * Virtual dtor...
         */
        virtual ~iLearner() { }

    public:

        /**
         * Functions of this signature know how to read a specific learner
         * subclass from a stream.
         */
        typedef iLearner* (*newLearnerFunc)(iReadable* in);

        /**
         * Use this method in each learner subclass to register itself so
         * that it can be read from a stream by newLearnerFromStream().
         */
        static bool registerLearnerFuncWithHeaderId(newLearnerFunc func, u32 headerId);

        /**
         * Call this to read a learner from a stream. The specific learner subclass
         * that is built will not be known by the caller, but that's the beauty
         * of it! For this to work, each learner subclass must register itself
         * by calling registerLearnerFuncWithHeaderId() before you call newLearnerFromStream().
         */
        static iLearner* newLearnerFromStream(iReadable* in);

        /**
         * Writes the given learner to the stream.
         */
        static void writeLearnerToStream(iLearner* learner, iWritable* out);
};


}   // namespace ml2


#endif  // __ml2_iLearner_h__
