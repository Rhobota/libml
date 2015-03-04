// #if NDEBUG
// #include "../ml/Eigen/Core"
// #else
// #define NDEBUG 1                   // <-- comment-out these two lines if you need to debug
// #include "../ml/Eigen/Core"        //     stuff, especially if your program is crashing
// #undef NDEBUG                      // <-- somewhere inside Eigen doing so will help a lot
// #endif


/*
 * Uncomment the following two lines to ensure that no heap allocations
 * are happening without you knowing about it. (And then compile and
 * run your code and hope that no exceptions are thrown or asserts are
 * hit.)
 */
//#define EIGEN_NO_MALLOC 1
//#define EIGEN_DONT_PARALLELIZE 1


/*
 * To help not have heap allocations, we're increasing this value ourselves.
 * The Eigen default is lower.
 */
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000


/*
 * The Eigen include file:
 */
#include "../ml/Eigen/Core"


/*
 * These are typedefs we'll use a lot:
 */
typedef Eigen::Matrix< ml2::fml, Eigen::Dynamic, Eigen::Dynamic > Mat;
typedef Eigen::Map< Mat > Map;
typedef Eigen::Map< const Mat > MapConst;


