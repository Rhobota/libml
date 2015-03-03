#ifndef __ml2_common_h__
#define __ml2_common_h__


#include <ml2/rhocompat.h>

#include <rho/img/tImage.h>

#include <cmath>
#include <vector>


namespace ml2
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
// The logistic function:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

fml logistic_function(fml z);
fml derivative_of_logistic_function(fml z);
fml inverse_of_logistic_function(fml y);
fml logistic_function_min();
fml logistic_function_max();


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// The hyperbolic function:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

fml hyperbolic_function(fml z);
fml derivative_of_hyperbolic_function(fml z);
fml inverse_of_hyperbolic_function(fml y);
fml hyperbolic_function_min();
fml hyperbolic_function_max();


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


}    // namespace ml2


#endif   // __ml2_common_h__
