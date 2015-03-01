#ifndef __ml2_common_h__
#define __ml2_common_h__


#include <ml2/rhocompat.h>

#include <vector>


namespace ml2
{


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


/**
 * The logistic function:
 */
fml logistic_function(fml z);
fml derivative_of_logistic_function(fml z);
fml inverse_of_logistic_function(fml y);
fml logistic_function_min();
fml logistic_function_max();


/**
 * The hyperbolic function:
 */
fml hyperbolic_function(fml z);
fml derivative_of_hyperbolic_function(fml z);
fml inverse_of_hyperbolic_function(fml y);
fml hyperbolic_function_min();
fml hyperbolic_function_max();


}    // namespace ml2


#endif   // __ml2_common_h__
