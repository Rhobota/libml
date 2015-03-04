#include <ml2/iLearner.h>


namespace ml2
{


static std::map<u32, iLearner::newLearnerFunc>* gNewLearnerFuncs = NULL;


bool iLearner::registerLearnerFuncWithHeaderId(newLearnerFunc func, u32 headerId)
{
    if (gNewLearnerFuncs == NULL)
        gNewLearnerFuncs = new std::map<u32, iLearner::newLearnerFunc>();
    if (gNewLearnerFuncs->find(headerId) == gNewLearnerFuncs->end())
    {
        (*gNewLearnerFuncs)[headerId] = func;
        //std::cout << "Registered learner with header id: " << headerId << std::endl;
        return true;
    }
    else
    {
        throw eRuntimeError("Two learners have registered with the same header id!");
    }
}


iLearner* iLearner::newLearnerFromStream(iReadable* in)
{
    u32 headerId;
    rho::unpack(in, headerId);

    if (gNewLearnerFuncs == NULL || gNewLearnerFuncs->find(headerId) == gNewLearnerFuncs->end())
    {
        throw eRuntimeError("No newLearnerFunc registered for that header id. Call iLearner::registerLearnerFuncWithHeaderId() with this header id before you call iLearner::newLearnerFromStream().");
    }

    newLearnerFunc func = (*gNewLearnerFuncs)[headerId];
    return func(in);
}


void iLearner::writeLearnerToStream(iLearner* learner, iWritable* out)
{
    u32 headerId = learner->headerId();
    rho::pack(out, headerId);
    learner->pack(out);
}


}   // namespace ml2
