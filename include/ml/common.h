#ifndef __ml_common_h__
#define __ml_common_h__


#include <ml/rhocompat.h>
#include <ml/iLearner_pre.h>

#include <rho/img/tImage.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <utility>
#include <vector>


namespace ml
{


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Typedefs:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * We can easily switch floating point precisions with the
 * following typedef. This effects the precision of the
 * input/output/target examples and the internal precision
 * of the layers.
 */
typedef f32 fml;
#define FML(x) x ## f   // <-- used to append 'f' to the end of fml literals

/**
 * This object will be used to denote input to the learner as
 * well as output from the learner. Both input/output to/from
 * this learner are vector data.
 *
 * This typedef makes it easier to represent several input examples
 * or several target examples without having to explicitly declare
 * vectors of vectors of floats.
 */
typedef std::vector<fml> tIO;


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// IO manipulation tools:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * Turns the integer value into an example that can be trained-on.
 * The returned training example has 'numDimensions' number of
 * dimensions, where one dimension is set to 1.0, and all others
 * are set to 0.0. The high dimension's index is given by
 * 'highDimension'.
 *
 * This is useful for creating the target vector when training
 * a classifier.
 */
tIO examplify(u32 highDimension, u32 numDimensions);

/**
 * Does the opposite operation as the above examplify() function.
 * It does so by determining which dimension has the highest
 * value, and returns the index to that dimension.
 *
 * If 'error' is not NULL, the squared error between the given output
 * and the "correct" output is calculated and stored in 'error'.
 * The "correct" output is obtained by calling the examplify()
 * function above. The assumption is made that the returned
 * index for the highest dimension is correct, thus the method
 * calculates the squared error between the given output and
 * the "correct" output.
 *
 * This is useful for evaluating the output of a classifier.
 */
u32 un_examplify(const tIO& output, fml* error = NULL);

/**
 * Turns the given image into an example that can be trained-on.
 */
tIO examplify(const img::tImage* image);

/**
 * Generates an image representation of the given tIO object, 'io'.
 *
 * If 'io' should be interpreted as an RGB image, set 'color'
 * to true. If the 'io' should be interpreted as a grey image,
 * set 'color' to false.
 *
 * You must specify the 'width' of the generated image. The
 * height will be derived by this function.
 *
 * If 'color' is false, then the green channel of the output image
 * is used to indicate positive values in 'io' and the red channel
 * of the output image is used to indicate negative values in 'io'.
 *
 * If 'color' is true, the trick above cannot be used because we
 * need each channel of the output image to represent itself. In
 * this case, the 'absolute' parameter is used to help determine
 * how to generate the output image.
 *
 * If 'absolute' is set to true, the absolute value of 'io'
 * will be used when producing the image. Otherwise, the relative
 * values will be used to produce the image (meaning that values
 * equal to zero will not be black if there are any negative values
 * in 'io').
 *
 * If the data has a finite range, you can specify that range
 * so that un_examplify() can create an image that respects
 * it. Otherwise, un_examplify() will use the min and max
 * of the data itself as the range so that the generated
 * image uses the full range of color.
 *
 * The generated image is stored in 'dest'.
 */
void un_examplify(const tIO& io, bool color, u32 width,
                  bool absolute, img::tImage* dest,
                  const fml* minValue = NULL, const fml* maxValue = NULL);

/**
 * Z-score a set of input examples.
 *
 * Z-scoring is transforming the data so that each dimension's mean is zero
 * and its standard deviation is one.
 *
 * The 'dStart' and 'dEnd' indices let you specify which dimensions of the
 * data you are interesting in zscoring. The dimensions that will be zscored
 * are [dStart, dEnd). Note that by default all dimensions will be zscored.
 */
void zscore(std::vector<tIO>& inputs, u32 dStart=0, u32 dEnd=0xFFFFFFFF);

/**
 * Z-score the training set, and z-score the test set to match.
 *
 * Z-scoring is transforming the data so that each dimension's mean is zero
 * and its standard deviation is one.
 */
void zscore(std::vector<tIO>& trainingInputs,
            std::vector<tIO>& testInputs);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Error measures:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * Calculates and returns the squared error between the given
 * output and the given target.
 */
fml squaredError(const tIO& output, const tIO& target);

/**
 * Calculates the mean squared error between each output/target
 * pair.
 */
fml meanSquaredError(const std::vector<tIO>& outputs,
                     const std::vector<tIO>& targets);

/**
 * Calculates and returns the cross-entropy cost between the given
 * output and the given target.
 */
fml crossEntropyCost(const tIO& output, const tIO& target);

/**
 * Calculates the average cross-entropy cost between each output/target
 * pair.
 */
fml crossEntropyCost(const std::vector<tIO>& outputs,
                     const std::vector<tIO>& targets);

/**
 * Calculates the root-mean-squared error of the output/target
 * pairs.
 *
 * (Note: This error function does not have a single output/target
 *        pair version because that wouldn't not make sense for this
 *        particular way of measuring error.)
 */
fml rmsError(const std::vector<tIO>& outputs,
             const std::vector<tIO>& targets);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Confusion matrix tools:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * This typedef makes creating a confusion matrix easier.
 */
typedef std::vector< std::vector<u32> > tConfusionMatrix;

/**
 * Creates a confusion matrix for the given output/target
 * pairs. For each output/target pair, un_examplify() is
 * called twice (once for the output and once for the target),
 * and the corresponding entry in the confusion matrix is
 * incremented.
 */
void buildConfusionMatrix(const std::vector<tIO>& outputs,
                          const std::vector<tIO>& targets,
                                tConfusionMatrix& confusionMatrix);

/**
 * Same as buildConfusionMatrix() above, but this function
 * does not simply count the entries in each cell of the
 * confusion matrix, it actually draws the input examples in
 * the cells of the confusion matrix! This gives you a
 * visual representation of the confusion matrix.
 *
 * The inputs are assumed to be images, for how else could
 * we draw them!? The inputs are transformed into images
 * by calling un_examplify() on them. See the comments
 * of un_examplify() for details on 'color', 'width', and
 * 'absolute'.
 *
 * The resulting image is stored in 'dest'.
 *
 * The width of each cell is calculated by multiplying the
 * 'width' by 'cellWidthMultiplier'. So the width of 'dest'
 * will be 'width*cellWidthMultiplier*<num_classes>'
 */
void buildVisualConfusionMatrix(const std::vector<tIO>& inputs,
                                bool color, u32 width, bool absolute,
                                const std::vector<tIO>& outputs,
                                const std::vector<tIO>& targets,
                                      img::tImage* dest,
                                u32 cellWidthMultiplier = 5);

/**
 * Prints the confusion matrix in a pretty format.
 */
void print(const tConfusionMatrix& confusionMatrix, std::ostream& out);

/**
 * Calculates the error rate for the given confusion matrix.
 *
 * Works for any confusion matrix.
 */
f64  errorRate(const tConfusionMatrix& confusionMatrix);

/**
 * Calculates the accuracy for the given confusion matrix.
 *
 * Works for any confusion matrix.
 */
f64  accuracy(const tConfusionMatrix& confusionMatrix);

/**
 * Calculates the precision of the confusion matrix.
 *
 * Only works for confusion matrices that have true/false
 * dimensions (aka, confusion matrices that are two-by-two).
 */
f64  precision(const tConfusionMatrix& confusionMatrix);

/**
 * Calculates the recall of the confusion matrix.
 *
 * Only works for confusion matrices that have true/false
 * dimensions (aka, confusion matrices that are two-by-two).
 */
f64  recall(const tConfusionMatrix& confusionMatrix);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training and visualization low-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class iInputTargetGenerator
{
    public:

        /**
         * This method is called when more input/target pairs are needed.
         *
         * The requester passes 'count' as the maximum it wants to receive.
         * You should return 'count' input/target pairs unless you've reached
         * the end of this epoch (in which case you may return less). If you
         * return less than 'count', be sure to resize 'fillmeInputs' and
         * 'fillmeTargets' so that they contains the correct number of
         * input/target pairs.
         *
         * If there are no input/target pairs left to return, be sure to
         * resize 'fillmeInputs' and 'fillmeTargets' to zero length.
         */
        virtual void generate(u32 count, std::vector<tIO>& fillmeInputs,
                                         std::vector<tIO>& fillmeTargets) = 0;

        /**
         * This method is the same as generate() above, except it is used
         * when the target values are not needed.
         */
        virtual void generate(u32 count, std::vector<tIO>& fillmeInputs) = 0;

        /**
         * This method is called between epochs. When called, you should shuffle
         * your data (if appropriate), to prepare for the next epoch to begin.
         */
        virtual void shuffle() = 0;

        /**
         * This method is called to restart the current epoch. This method can
         * also be used to train on the just-completed epoch without doing a shuffle,
         * in the case that (for some reason) you don't want to shuffle your data
         * between epochs.
         */
        virtual void restart() = 0;

        virtual ~iInputTargetGenerator() { }
};

class iOutputCollector
{
    public:

        /**
         * This method is called after every batch is evaluated.
         * It contains the inputs, targets, and outputs of the batch.
         */
        virtual void receivedOutput(const std::vector<tIO>& inputs,
                                    const std::vector<tIO>& targets,
                                    const std::vector<tIO>& outputs) = 0;

        virtual ~iOutputCollector() { }
};

class iOutputPerformanceEvaluator : public iOutputCollector
{
    public:

        /**
         * Returns the human-readable name of the method being used
         * to evaluate performance here.
         */
        virtual std::string evaluationMethodName() = 0;

        /**
         * Call this method after this collector has collected all
         * its inputs/targets/outputs.
         *
         * This method will use its internal evaluation method, and
         * it will return the result.
         */
        virtual f64 calculatePerformance() = 0;

        /**
         * Indicates whether or not positive (i.e. increasing) values
         * of the performance metric returned by calculatePerformance()
         * are good (i.e. indicates better performance).
         *
         * E.g. If the performance metric is AUC, this method should
         * return true.
         *
         * E.g. If the performance metric is average mean squared
         * error, this method should return false. Because for the
         * squared error metric, decreasing values indicate better
         * performance.
         */
        virtual bool isPositivePerformanceGood() = 0;

        /**
         * This method will reset this object so that you can use it
         * fresh to collect more inputs/targets/outputs and then
         * calculate the performance on that new input. Basically,
         * this method could also be called "forget".
         */
        virtual void reset() = 0;
};

/**
 * This class assumes you're doing classification. Any number of classes is fine.
 * It calculates the classicition error rate.
 */
class tOutputPerformanceEvaluatorClassificationError : public iOutputPerformanceEvaluator
{
    public:

        tOutputPerformanceEvaluatorClassificationError()
        {
            reset();
        }

        ///////////////////////////////////////////////////////////////////////
        // iOutputCollector interface:
        ///////////////////////////////////////////////////////////////////////

        void receivedOutput(const std::vector<tIO>& inputs,
                            const std::vector<tIO>& targets,
                            const std::vector<tIO>& outputs)
        {
            assert(inputs.size() == targets.size());
            assert(targets.size() == outputs.size());

            for (size_t i = 0; i < outputs.size(); i++)
            {
                assert(outputs[i].size() == targets[i].size());
                assert(outputs[i].size() > 0);

                u32 outputClass = un_examplify(outputs[i]);
                u32 targetClass = un_examplify(targets[i]);

                if (outputClass != targetClass)
                    ++m_errors;
                ++m_total;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // iOutputPerformanceEvaluator interface:
        ///////////////////////////////////////////////////////////////////////

        std::string evaluationMethodName()
        {
            return "Classification Error Rate";
        }

        f64 calculatePerformance()
        {
            if (m_total == 0)
                throw eRuntimeError("Cannot calculate performance on zero examples.");
            f64 error = m_errors;
            error /= m_total;
            return error;
        }

        bool isPositivePerformanceGood()
        {
            return false;
        }

        void reset()
        {
            m_errors = 0;
            m_total = 0;
        }

    private:

        u32 m_errors;
        u32 m_total;
};

/**
 * This class assumes you're doing binary classification, and it is used
 * to calculate the Area Under the ROC Curve (AUC).
 *
 * It works in two modes (it automatically detects each mode).
 *
 * The first mode is where the model outputs a single fml value in [0,1].
 * As you hopefully expect, that value indicates a probability of being
 * in the positive class. I.e. 0 is a sure negative case and 1 is sure positive
 * case.
 *
 * The second mode is where the model outputs two fml values. The first value
 * is the probability of a negative case, and the second value is the probability
 * of a positive case. Typically these two values will sum to 1, but the code below
 * doesn't require that to be true.
 */
class tOutputPerformanceEvaluatorAUC : public iOutputPerformanceEvaluator
{
    public:

        /////////////////////////////////////////////////////////////////////////////////////////////
        // iOutputCollector interface:
        /////////////////////////////////////////////////////////////////////////////////////////////

        void receivedOutput(const std::vector<tIO>& inputs,
                            const std::vector<tIO>& targets,
                            const std::vector<tIO>& outputs)
        {
            assert(inputs.size() == targets.size());
            assert(targets.size() == outputs.size());

            for (size_t i = 0; i < targets.size(); i++)
            {
                assert(targets[i].size() == outputs[i].size());
                assert(outputs[i].size() == 1 || outputs[i].size() == 2);

                if (outputs[i].size() == 1)
                {
                    u32 target = (u32) round(targets[i][0]);
                    fml output = outputs[i][0];
                    m_predictions.push_back(std::make_pair(output, target));
                }
                else  // outputs[i].size() == 2
                {
                    u32 target = un_examplify(targets[i]);
                    fml output = outputs[i][1] / (outputs[i][0] + outputs[i][1]);
                    m_predictions.push_back(std::make_pair(output, target));
                }
            }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        // iOutputPerformanceEvaluator interface:
        /////////////////////////////////////////////////////////////////////////////////////////////

        std::string evaluationMethodName()
        {
            return "AUC";
        }

        f64 calculatePerformance()
        {
            std::sort(m_predictions.begin(), m_predictions.end());

            u32 numPos = 0;
            for (size_t i = 0; i < m_predictions.size(); i++)
                if (m_predictions[i].second == 1)
                    numPos += 1;

            u32 tp = numPos;
            u64 coveredArea = 0;
            u64 maxArea = 0;
            for (size_t i = 0; i < m_predictions.size(); i++)
            {
                if (m_predictions[i].second == 1)
                    tp -= 1;
                else
                {
                    coveredArea += tp;
                    maxArea += numPos;
                }
            }

            return ((f64)coveredArea) / ((f64)maxArea);
        }

        bool isPositivePerformanceGood()
        {
            return true;
        }

        void reset()
        {
            m_predictions.clear();
        }

    private:

        std::vector< std::pair<fml,u32> > m_predictions;
};

class iTrainObserver
{
    public:

        /**
         * This method is called by the train() function below after
         * update() has been called on the given learner. The inputs
         * and targets passed are from the most recent batch used
         * for training.
         *
         * This method should return true if all is well and training
         * should continue. It should return false if the training
         * process should halt. This is useful if you need to cancel
         * training due to user input, or something like that.
         */
        virtual bool didUpdate(iLearner* learner, const std::vector<tIO>& inputs,
                                                  const std::vector<tIO>& targets) = 0;

        virtual ~iTrainObserver() { }
};

/**
 * This function trains the leaner for one epoch,
 * calling the training observer after each batch has been
 * processed by the learner. This function returns true if
 * the training process completed fully, and it returns false
 * if the training observer indicated that training should
 * halt early.
 *
 * Use this for training on the training-set.
 */
bool train(iLearner* learner, iInputTargetGenerator* generator,
                              u32 batchSize,
                              iTrainObserver* trainObserver = NULL);

/**
 * This function uses the learner to evaluate every input
 * examples from one epoch's-worth of input obtained via
 * the generator.
 * Use the collector to collect the output of each input.
 * Inside the collector, you can do whatever you want with
 * the results. E.g. Create a confusion matrix, or just calculate
 * accuracy, or whatever.
 *
 * Use this for evaluating how the learner is doing on the
 * training-set AND test-set. Knowing how the learner is doing
 * on each is vital for determining bias and variance of your
 * model.
 */
void evaluate(iLearner* learner, iInputTargetGenerator* generator,
                                 iOutputCollector* collector,
                                 u32 batchSize=1000);

/**
 * Creates a visual of the learner processing the example provided.
 * The visual is stored as an image in 'dest'.
 */
void visualize(iLearner* learner, const tIO& example,
               bool color, u32 width, bool absolute,
               img::tImage* dest);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training medium-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class iEZTrainObserver : public iTrainObserver
{
    public:

        /**
         * This method is called by the ezTrain() function below after
         * a full epoch of training has been done on the given learner.
         * It should return true if all is well and training should continue.
         * It should return false if the training process should halt. This
         * is useful if you need to cancel training due to user input, or
         * if you detect that the learner has been trained enough and is ready
         * to be used.
         */
        virtual bool didFinishEpoch(iLearner* learner,
                                    u32 epochsCompleted,
                                    f64 epochTrainTimeInSeconds,
                                    f64 trainingSetPerformance,
                                    f64 testSetPerformance,
                                    bool positivePerformanceIsGood) = 0;

        /**
         * This method is called after training completes, meaning that
         * didFinishEpoch() will not be called anymore.
         */
        virtual void didFinishTraining(iLearner* learner,
                                       u32 epochsCompleted,
                                       f64 trainingTimeInSeconds) = 0;
};

/**
 * This function trains the leaner on the given training set,
 * and tests the learner on the given test set. It trains
 * for as many epochs are needed by calling the train()
 * function above to train the learner on each epoch. This
 * function takes a train observer which it notifies (if not
 * null) every 'evaluationInterval' epochs with the most recent
 * training results. This function will not return until the
 * observer indicates that training can halt.
 *
 * This function returns the number of epochs of training which
 * were completed.
 *
 * This function is intended to replace calling train() over-and-
 * over in most application where straight-forward training is needed.
 */
u32  ezTrain(iLearner* learner, iInputTargetGenerator* trainingSetGenerator,
                                iInputTargetGenerator* testSetGenerator,
                                u32 batchSize, u32 evaluationInterval,
                                iEZTrainObserver* trainObserver = NULL);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training high-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class tSmartStoppingWrapper : public iEZTrainObserver
{
    public:

        /**
         * This class defines an early stopping condition for training
         * a learner.
         *
         * It guarantees that the learner will be trained for at least
         * 'minEpochs' number of epochs, even if no progress is seen.
         *
         * It guarantees that the learner will be trained for at most
         * 'maxEpochs' number of epochs, even if progress is seen the
         * entire time.
         *
         * The algorithm respects significant improvements in performance
         * and may increase the allowed training time when a significant
         * improvement is encountered. If performance increases by
         * 'significantThreshold' or more, the increase is considered
         * significant. When a significant improvement happens, the allowed
         * training time is extended to be at least the current duration of
         * training time multiplied by 'patienceIncrease'.
         *
         * This class wraps another iEZTrainObserver so that observers
         * can be decorated by objects of this type.
         */
        tSmartStoppingWrapper(u32 minEpochs=50,
                              u32 maxEpochs=1000,
                              f64 significantThreshold=0.005,   // <-- half a percent
                              f64 patienceIncrease=2.0,
                              iEZTrainObserver* wrappedObserver=NULL);

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector<tIO>& inputs,
                                          const std::vector<tIO>& targets);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
                            u32 epochsCompleted,
                            f64 epochTrainTimeInSeconds,
                            f64 trainingSetPerformance,
                            f64 testSetPerformance,
                            bool positivePerformanceIsGood);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               f64 trainingTimeInSeconds);

    private:

        void m_reset();

    private:

        const u32 m_minEpochs;
        const u32 m_maxEpochs;
        const f64 m_significantThreshold;
        const f64 m_patienceIncrease;

        iEZTrainObserver * const m_obs;

        f64 m_bestFoundTestSetPerformance;
        u32 m_allowedEpochs;
};


class tBestRememberingWrapper : public iEZTrainObserver
{
    public:

        /**
         * This class wraps a iEZTrainObserver to add the ability to remember the
         * learner which performed best on the test set during training. This
         * allows you to identify which point in the learning process gave you
         * the best generalization error estimate, and then to recreate that point
         * at the end of training.
         */
        tBestRememberingWrapper(iEZTrainObserver* wrappedObserver=NULL);

        void reset();

        u32 bestAfterEpochsCompleted() const;
        f64 bestTestSetPerformance()   const;

        iLearner* newBestLearner() const;   // <-- caller must call delete on this when finished with it

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector<tIO>& inputs,
                                          const std::vector<tIO>& targets);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
                            u32 epochsCompleted,
                            f64 epochTrainTimeInSeconds,
                            f64 trainingSetPerformance,
                            f64 testSetPerformance,
                            bool positivePerformanceIsGood);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               f64 trainingTimeInSeconds);

    private:

        u32 m_bestAfterEpochsCompleted;
        f64 m_bestTestSetPerformance;

        tByteWritable m_serializedLearner;

        iEZTrainObserver * const m_obs;
};


class tLoggingWrapper : public tBestRememberingWrapper
{
    public:

        /**
         * This class provides a wrapper around another iEZTrainObserver
         * to decorate it with logging ability. Note that this class
         * extends tBestRememberingWrapper, so if you need the functionality
         * of tBestRememberingWrapper you do not need to add it to the
         * decoration chain yourself because it comes free when you use this
         * class.
         *
         * Learner performance is logged every epoch to a human-readable log
         * file and to a simplified data log file. Every log file produced by
         * this class is prefixed with the 'fileprefix' string specified
         * to the constructor.
         *
         * Every 'logInterval' number of epochs, the learner itself is
         * serialized to a file.
         */
        tLoggingWrapper(u32 logInterval,
                        iEZTrainObserver* wrappedObserver = NULL,
                        std::string fileprefix=std::string());

        ~tLoggingWrapper();

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector<tIO>& inputs,
                                          const std::vector<tIO>& targets);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
                            u32 epochsCompleted,
                            f64 epochTrainTimeInSeconds,
                            f64 trainingSetPerformance,
                            f64 testSetPerformance,
                            bool positivePerformanceIsGood);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               f64 trainingTimeInSeconds);

    private:

        const u32 m_logInterval;
        std::string m_fileprefix;

        std::ofstream m_logfile;
        std::ofstream m_datafile;
};


}    // namespace ml


#endif   // __ml_common_h__
