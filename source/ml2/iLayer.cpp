#include <ml2/iLayer.h>


namespace ml2
{


static std::map<u32, iLayer::newLayerFunc> gNewLayerFuncs;


bool iLayer::registerLayerFuncWithHeaderId(newLayerFunc func, u32 headerId)
{
    if (gNewLayerFuncs.find(headerId) == gNewLayerFuncs.end())
    {
        gNewLayerFuncs[headerId] = func;
        return true;
    }
    else
    {
        throw eRuntimeError("Two layers have registered with the same header id!");
    }
}


iLayer* iLayer::newLayerFromStream(iReadable* in)
{
    u32 headerId;
    rho::unpack(in, headerId);

    if (gNewLayerFuncs.find(headerId) == gNewLayerFuncs.end())
    {
        throw eRuntimeError("No newLayerFunc registered for that header id. Call iLayer::registerLayerFuncWithHeaderId() with this header id before you call iLayer::newLayerFromStream().");
    }

    newLayerFunc func = gNewLayerFuncs[headerId];
    return func(in);
}


void iLayer::writeLayerToStream(iLayer* layer, iWritable* out)
{
    u32 headerId = layer->headerId();
    rho::pack(out, headerId);
    layer->pack(out);
}


}   // namespace ml2
