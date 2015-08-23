#ifndef __ml_common_h__
#define __ml_common_h__


#include <ml/rhocompat.h>
#include <ml/iLearner_pre.h>

#include <rho/img/tImage.h>

#include <cmath>
#include <fstream>
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
 * If 'error' is not NULL, the std squared error between the given
 * output and the "correct" output is calculated and stored in 'error'.
 * The "correct" output is obtained by calling the examplify()
 * function above. The assumption is made that the returned
 * index for the highest dimension is correct, thus the method
 * calculates the standard error between the given output and
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
 * Calculates and returns the standard squared error between the given
 * output and the given target.
 */
fml standardSquaredError(const tIO& output, const tIO& target);

/**
 * Calculates the average standard squared error between each output/target
 * pair.
 */
fml standardSquaredError(const std::vector<tIO>& outputs,
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
         * Return true if there is still more input to get in this epoch.
         * Return false if the last of the current epoch has been returned
         * by this call.
         *
         * The requester passes 'count' as the maximum it wants to receive.
         * You should return 'count' input/target pairs unless you've reached
         * the end of this epoch (in which case you may return less). If you
         * return less than 'count', be sure to resize 'fillme' so that it
         * contains the correct number of input/target pairs.
         */
        virtual bool generate(u32 count, std::vector< std::pair<tIO,tIO> >& fillme) = 0;

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
        virtual void receivedOutput(const std::vector< std::pair<tIO,tIO> >& inputTargetPairs,
                                    const std::vector<               tIO  >& outputs) = 0;

        virtual ~iOutputCollector() { }
};

class iTrainObserver
{
    public:

        /**
         * This method is called by the train() function below after
         * update() has been called on the given learner. It should
         * return true if all is well and training should continue.
         * It should return false if the training process should
         * halt. This is useful if you need to cancel training
         * due to user input, or something like that.
         */
        virtual bool didUpdate(iLearner* learner, const std::vector< std::pair<tIO,tIO> >& mostRecentBatch) = 0;

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
                                    f64 testSetPerformance) = 0;

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
 * null) after each epoch with the most recent training results.
 * This function will not return until the observer indicates
 * that training can halt.
 *
 * This function returns the number of epochs of training which
 * were completed.
 *
 * This function is intended to replace calling train() over-and-
 * over in most application where straight-forward training is needed.
 */
u32  ezTrain(iLearner* learner, iInputTargetGenerator* trainingSetGenerator,
                                iInputTargetGenerator* testSetGenerator,
                                u32 batchSize,
                                iEZTrainObserver* trainObserver = NULL);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training high-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

enum nPerformanceAttribute
{
    kClassificationErrorRate      = 1,
    kOutputErrorMeasure           = 2
};

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
         *
         * You can choose which performance measure is used to determine
         * when to stop with the last parameter.
         */
        tSmartStoppingWrapper(u32 minEpochs=50,
                              u32 maxEpochs=1000,
                              f64 significantThreshold=0.005,   // <-- half a percent
                              f64 patienceIncrease=2.0,
                              iEZTrainObserver* wrappedObserver=NULL,
                              nPerformanceAttribute performanceAttribute=kOutputErrorMeasure);

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector< std::pair<tIO,tIO> >& mostRecentBatch);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
                            u32 epochsCompleted,
                            u32 foldIndex, u32 numFolds,
                            const std::vector< tIO >& trainInputs,
                            const std::vector< tIO >& trainTargets,
                            const std::vector< tIO >& trainOutputs,
                            const tConfusionMatrix& trainCM,
                            const std::vector< tIO >& testInputs,
                            const std::vector< tIO >& testTargets,
                            const std::vector< tIO >& testOutputs,
                            const tConfusionMatrix& testCM,
                            f64 epochTrainTimeInSeconds);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds);

    private:

        void m_reset();

    private:

        const u32 m_minEpochs;
        const u32 m_maxEpochs;
        const f64 m_significantThreshold;
        const f64 m_patienceIncrease;

        iEZTrainObserver * const m_obs;

        f64 m_bestTestErrorYet;
        u32 m_allowedEpochs;

        nPerformanceAttribute m_performanceAttribute;
};


class tBestRememberingWrapper : public iEZTrainObserver
{
    public:

        /**
         * This class wraps a iEZTrainObserver to add the ability to
         * remember the learner and the output values which performed
         * best on the test set during training. This allows you to identify
         * which point in the learning process gave you the best generalization
         * error estimate, and then to report that point at the end of training.
         *
         * Using 'performanceAttribute', you can choose which performance measure
         * is used for determining the best network observed.
         */
        tBestRememberingWrapper(iEZTrainObserver* wrappedObserver=NULL,
                                nPerformanceAttribute performanceAttribute=kClassificationErrorRate);

        void reset();

        u32 bestTestEpochNum()  const;
        f64 bestTestError() const;

        const std::vector<tIO>& bestTestOutputs() const;
        const tConfusionMatrix& bestTestCM()      const;
        const tConfusionMatrix& matchingTrainCM() const;

        void newBestLearner(iLearner*& learner)   const;   // <-- caller must "delete learner;" when finished with it

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector< std::pair<tIO,tIO> >& mostRecentBatch);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
                            u32 epochsCompleted,
                            u32 foldIndex, u32 numFolds,
                            const std::vector< tIO >& trainInputs,
                            const std::vector< tIO >& trainTargets,
                            const std::vector< tIO >& trainOutputs,
                            const tConfusionMatrix& trainCM,
                            const std::vector< tIO >& testInputs,
                            const std::vector< tIO >& testTargets,
                            const std::vector< tIO >& testOutputs,
                            const tConfusionMatrix& testCM,
                            f64 epochTrainTimeInSeconds);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds);

    private:

        u32 m_bestTestEpochNum;
        f64 m_bestTestErrorRate;

        std::vector<tIO> m_bestTestOutputs;
        tConfusionMatrix m_bestTestCM;
        tConfusionMatrix m_matchingTrainCM;

        tByteWritable m_serializedLearner;

        iEZTrainObserver * const m_obs;

        nPerformanceAttribute m_performanceAttribute;
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
         * Error rates are logged every epoch to a human-readable log file
         * and to a simplified data log file. Every log file produced by
         * this class is prefixed with the 'fileprefix' string specified
         * to the constructor.
         *
         * Every 'logInterval' number of epochs, the learner itself is
         * serialized to a file, and visualization of the learner and its
         * progress are also saved if 'logVisuals' is true.
         *
         * See un_examplify() for a description of 'isInputImageColor',
         * 'inputImageWidth', and 'shouldDisplayAbsoluteValues'. These
         * parameters are needed for creating the visualization which occur
         * every 'logInterval' number of epochs. These parameters are only
         * relevant if 'logVisuals' is true.
         *
         * See buildConfusionMatrix() for a description of 'cellWidthMultiplier'.
         * This parameter is only relevant if 'logVisuals' is true.
         *
         * The visualizations of the learner are useful when the learner is
         * processing image data. If the input data is not image data,
         * the visuals are not as meaningful, and in some cases cannot be
         * produced anyway because of data alignment issues. You use
         * 'logVisuals' to turn off the logging of these visualizations.
         *
         * See tBestRememberingWrapper for a description of 'performanceAttribute'.
         *
         * This logging wrapper works well when used with both versions of
         * ezTrain() above. That is, it works for normal train/test sets,
         * and it works for x-fold cross-validation sets.
         *
         * If 'accumulateFoldIO' is set, a few things will happen:
         *     1. If using more than one fold, the test inputs/targets/outputs
         *        of each fold will be accumulated and will be displayed
         *        at the end of learning (such as the accumulated CM and
         *        the accumulated error rate and an accumulated visual CM).
         *        Note: Accumulating the test inputs/targets/outputs like
         *        this is memory intensive when the training set is large.
         *     2. A binary log file will be written that contains
         *        the accumulated test inputs/targets/outputs for each fold
         *        (this will happen even if there is only one fold). This
         *        binary file is useful for doing post processing on the
         *        outputs, such as evaluating the outputs using a different
         *        cost function, or trying to combine or morph the outputs
         *        to decrease the error in some way.
         *        Note: This binary log file will be very large if the training
         *        set is large.
         */
        tLoggingWrapper(u32 logInterval, bool isInputImageColor,
                        u32 inputImageWidth, bool shouldDisplayAbsoluteValues,
                        u32 cellWidthMultiplier = 5,
                        iEZTrainObserver* wrappedObserver = NULL,
                        bool accumulateFoldIO = false,
                        bool logVisuals = true,
                        std::string fileprefix=std::string(),
                        nPerformanceAttribute performanceAttribute=kClassificationErrorRate);

        ~tLoggingWrapper();

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector< std::pair<tIO,tIO> >& mostRecentBatch);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
                            u32 epochsCompleted,
                            u32 foldIndex, u32 numFolds,
                            const std::vector< tIO >& trainInputs,
                            const std::vector< tIO >& trainTargets,
                            const std::vector< tIO >& trainOutputs,
                            const tConfusionMatrix& trainCM,
                            const std::vector< tIO >& testInputs,
                            const std::vector< tIO >& testTargets,
                            const std::vector< tIO >& testOutputs,
                            const tConfusionMatrix& testCM,
                            f64 epochTrainTimeInSeconds);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds);

    private:

        void m_save(std::string filebasename,
                    iLearner* learner,
                    const std::vector<tIO>& trainInputs,
                    const std::vector<tIO>& testInputs,
                    const std::vector<tIO>& testTargets,
                    const std::vector<tIO>& testOutputs);

    private:

        const u32  m_logInterval;
        const bool m_isColorInput;
        const u32  m_imageWidth;
        const bool m_absoluteImage;
        const u32  m_cellWidthMultiplier;

        const bool m_accumulateFoldIO;
        const bool m_logVisuals;

        std::string m_fileprefix;

        std::ofstream m_logfile;
        std::ofstream m_datafile;

        std::vector<tIO> m_accumTestInputs;     //
        std::vector<tIO> m_accumTestTargets;    //
        std::vector<tIO> m_accumTestOutputs;    //  These are only used if m_accumulateFoldIO is true
        tConfusionMatrix m_accumTestCM;         //
        tConfusionMatrix m_accumTrainCM;        //

        nPerformanceAttribute m_performanceAttribute;
};


}    // namespace ml


#endif   // __ml_common_h__
