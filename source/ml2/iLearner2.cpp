#include <ml2/iLearner.h>


namespace ml2
{


bool iLearner::registerLearnerFuncWithHeaderId(newLearnerFunc func, u32 headerId)
{
    // TODO
}


iLearner* iLearner::newLearnerFromStream(iReadable* in)
{
    // TODO
}


void iLearner::writeLearnerToStream(iLearner* learner, iWritable* out)
{
    // TODO
}


}   // namespace ml2
