// #if NDEBUG
// #include "../ml/Eigen/Core"
// #else
// #define NDEBUG 1                   // <-- comment-out these two lines if you need to debug
// #include "../ml/Eigen/Core"        //     stuff, especially if your program is crashing
// #undef NDEBUG                      // <-- somewhere inside Eigen doing so will help a lot
// #endif


//#define EIGEN_NO_MALLOC 1
#include "../ml/Eigen/Core"


typedef Eigen::Matrix< ml2::fml, Eigen::Dynamic, Eigen::Dynamic > Mat;
typedef Eigen::Map< Mat > Map;
typedef Eigen::Map< const Mat > MapConst;


