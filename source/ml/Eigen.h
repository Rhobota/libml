
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
//#define EIGEN_NO_MALLOC
//#define EIGEN_DONT_PARALLELIZE


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
#include "Eigen/Eigen/Core"


/*
 * These are typedefs we'll use a lot:
 */
#include <ml/common.h>
namespace ml
{
    typedef Eigen::Matrix< ml::fml, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > Mat;
    typedef Eigen::Map< Mat > Map;
    typedef Eigen::Map< const Mat > MapConst;

    typedef Eigen::Matrix< ml::fml, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatRowMajor;
    typedef Eigen::Map< MatRowMajor > MapRowMajor;
    typedef Eigen::Map< const MatRowMajor > MapRowMajorConst;

    typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;

    typedef Eigen::Map< Mat, Eigen::Unaligned, Stride > MapWithStride;
    typedef Eigen::Map< const Mat, Eigen::Unaligned, Stride > MapWithStrideConst;

    typedef Eigen::Map< MatRowMajor, Eigen::Unaligned, Stride > MapRowMajorWithStride;
    typedef Eigen::Map< const MatRowMajor, Eigen::Unaligned, Stride > MapRowMajorWithStrideConst;
}


/*
 * We will parallelize our own CPU code only if Eigen parallelizes its code.
 * That way there's only the one flag above to deal with turning on/off
 * parallelization in this library.
 */
#ifdef EIGEN_HAS_OPENMP
#define LIBML_HAS_OPENMP
#endif

