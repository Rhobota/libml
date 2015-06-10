#include <ml/kmeans.h>

#include <cassert>

using std::vector;


namespace ml
{


static
tIO s_randPoint(algo::iLCG& lcg, fml rmin, fml rmax, u32 dimensions)
{
    tIO v(dimensions);
    for (u32 i = 0; i < dimensions; i++)
    {
        v[i] = ((fml)lcg.next()) / ((fml)lcg.randMax());    // [0.0, 1.0]
        v[i] *= rmax-rmin;                                  // [0.0, rmax-rmin]
        v[i] += rmin;                                       // [rmin, rmax]
    }
    return v;
}


static
tIO s_calcCenter(const vector<tIO>& points, const vector<u32>& cluster)
{
    assert(cluster.size() > 0);

    tIO center(points[0].size(), 0.0);
    for (size_t i = 0; i < cluster.size(); i++)
        for (size_t j = 0; j < center.size(); j++)
            center[j] += points[cluster[i]][j];
    fml mul = FML(1.0) / ((fml)cluster.size());
    for (size_t j = 0; j < center.size(); j++)
        center[j] *= mul;
    return center;
}


static
fml s_distanceSquared(const tIO& p1, const tIO& p2)
{
    fml dist = 0.0;
    for (size_t i = 0; i < p1.size(); i++)
        dist += (p1[i]-p2[i]) * (p1[i]-p2[i]);
    return dist;
}


vector< vector<u32> > kmeans(const vector<tIO>& points, u32 k,
                             fml rmin, fml rmax, algo::iLCG& lcg,
                                   vector<tIO>& centers)
{
    if (points.size() == 0)
        throw eInvalidArgument("There must be points to cluster!");
    if (points[0].size() == 0)
        throw eInvalidArgument("There must be at least one dimension!");
    for (size_t i = 1; i < points.size(); i++)
        if (points[i].size() != points[0].size())
            throw eInvalidArgument("All data points must have the same dimensionality.");
    if (k == 0)
        throw eInvalidArgument("Clustering into zero clusters makes no sense.");
    if (rmin >= rmax)
        throw eInvalidArgument("rmin >= rmax makes no sense.");

    centers.resize(k);
    for (u32 i = 0; i < k; i++)
        centers[i] = s_randPoint(lcg, rmin, rmax, (u32)points[0].size());

    return kmeans(points, k, centers);
}


vector< vector<u32> > kmeans_pp(const vector<tIO>& points, u32 k,
                                algo::iLCG& lcg,
                                      vector<tIO>& centers)
{
    if (points.size() == 0)
        throw eInvalidArgument("There must be points to cluster!");
    if (points[0].size() == 0)
        throw eInvalidArgument("There must be at least one dimension!");
    for (size_t i = 1; i < points.size(); i++)
        if (points[i].size() != points[0].size())
            throw eInvalidArgument("All data points must have the same dimensionality.");
    if (k == 0)
        throw eInvalidArgument("Clustering into zero clusters makes no sense.");

    centers.resize(k);
    centers[0] = points[lcg.next() % points.size()];
    for (u32 i = 1; i < k; i++)
    {
        tIO cumDists(points.size());
        for (size_t d = 0; d < cumDists.size(); d++)
        {
            fml mind = s_distanceSquared(points[d], centers[0]);
            for (u32 j = 1; j < i; j++)
                mind = std::min(mind, s_distanceSquared(points[d], centers[j]));
            cumDists[d] = (d == 0) ? (mind) : (cumDists[d-1]+mind);
        }

        fml ran = ((fml)lcg.next()) / ((fml)lcg.randMax()) * cumDists.back();
        size_t d;
        for (d = 0; ran > cumDists[d]; d++) { }
        centers[i] = points[d];
    }

    return kmeans(points, k, centers);
}


vector< vector<u32> > kmeans(const vector<tIO>& points, u32 k,
                             vector<tIO>& centers)
{
    if (points.size() == 0)
        throw eInvalidArgument("There must be points to cluster!");
    if (points[0].size() == 0)
        throw eInvalidArgument("There must be at least one dimension!");
    for (size_t i = 1; i < points.size(); i++)
        if (points[i].size() != points[0].size())
            throw eInvalidArgument("All data points must have the same dimensionality.");
    if (k == 0)
        throw eInvalidArgument("Clustering into zero clusters makes no sense.");
    if ((u32)centers.size() != k)
        throw eInvalidArgument("You must supply k initial centers to this version of kmeans().");
    for (u32 i = 0; i < k; i++)
        if (centers[i].size() != points[0].size())
            throw eInvalidArgument("All initial centers must have the same dimensionality as the data points.");

    vector< vector<u32> > clusters(k);

    while (true)
    {
        vector< vector<u32> > newClusters(k);

        for (size_t i = 0; i < points.size(); i++)
        {
            fml dist = s_distanceSquared(points[i], centers[0]);
            u32 closest = 0;
            for (u32 c = 1; c < k; c++)
            {
                fml distHere = s_distanceSquared(points[i], centers[c]);
                if (distHere < dist)
                {
                    closest = c;
                    dist = distHere;
                }
            }
            newClusters[closest].push_back((u32)i);
        }

        for (u32 i = 0; i < k; i++)
            if (newClusters[i].size() > 0)
                centers[i] = s_calcCenter(points, newClusters[i]);
            // Else, what should I do? Leave it? Randomize a new center?

        if (clusters == newClusters)
            break;

        clusters = newClusters;
    }

    return clusters;
}


}   // namespace ml
