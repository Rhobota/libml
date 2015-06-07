
/*
 * Uncomment the following two lines to ensure that no heap allocations
 * are happening without you knowing about it. And then compile and
 * run your code and hope that no exceptions are thrown or asserts are
 * hit. Note that heap allocations may still happen if more memory is
 * required than what Eigen wants to put on the stack, even if you are
 * doing everything correctly on your end. To test this, be sure to use
 * small matrices with Eigen so that you can be sure that any heap
 * allocations that happen are in fact your fault. Related to this, see
 * the next comment.
 */
#define EIGEN_NO_MALLOC
#define EIGEN_DONT_PARALLELIZE


/*
 * To help not have heap allocations, we're increasing this value ourselves.
 * The Eigen default is lower.
 */
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000


/*
 * Uncomment the following line to put Eigen in no-asserts mode. Only
 * do this if you have tested a lot since your last changes. Having
 * Eigen in no-asserts mode will make it run slightly faster.
 */
//#define NDEBUG 1


/*
 * The Eigen include file:
 */
#include "Eigen/Core"


/*
 * These are typedefs we'll use a lot:
 */
#include <ml2/common.h>
namespace ml2
{
    typedef Eigen::Matrix< ml2::fml, Eigen::Dynamic, Eigen::Dynamic > Mat;
    typedef Eigen::Map< Mat > Map;
    typedef Eigen::Map< const Mat > MapConst;
}

