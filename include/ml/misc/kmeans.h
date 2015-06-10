#ifndef __ml_kmeans_h__
#define __ml_kmeans_h__


#include <ml/rhocompat.h>
#include <ml/iLearner.h>

#include <rho/algo/tLCG.h>
#include <rho/types.h>

#include <vector>


namespace ml
{
namespace misc
{


/**
 * Runs kmeans on 'points' with 'k' centers.
 *
 * The centers will be randomly initialized
 * over the uniform distribution [rmin, rmax].
 * The given LCG will be used to initialize
 * the centers.
 *
 * The 'centers' variable is an out-param only.
 * The 'centers' variable will contain the
 * resulting centers when the function returns.
 *
 * An array of 'k' clusters is returned, where
 * each cluster is an array of indices into
 * 'points'.
 */
std::vector< std::vector<u32> > kmeans(const std::vector<tIO>& points, u32 k,
                                       fml rmin, fml rmax, algo::iLCG& lcg,
                                             std::vector<tIO>& centers);


/**
 * Runs kmeans++ on 'points' with 'k' centers.
 *
 * The given LCG will be used to randomly choose initial
 * centers, as determined by the kmeans++ algorithm.
 *
 * The 'centers' variable is an out-param only.
 * The 'centers' variable will contain the
 * resulting centers when the function returns.
 *
 * An array of 'k' clusters is returned, where
 * each cluster is an array of indices into
 * 'points'.
 */
std::vector< std::vector<u32> > kmeans_pp(const std::vector<tIO>& points, u32 k,
                                          algo::iLCG& lcg,
                                                std::vector<tIO>& centers);


/**
 * Runs kmeans on 'points' with 'k' centers.
 *
 * The 'centers' variable is an in/out param.
 * The initial centers are given by 'centers', and
 * the 'centers' variable will contain the
 * updated centers when the function returns.
 *
 * An array of 'k' clusters is returned, where
 * each cluster is an array of indices into
 * 'points'.
 */
std::vector< std::vector<u32> > kmeans(const std::vector<tIO>& points, u32 k,
                                             std::vector<tIO>& centers);


}   // namespace misc
}   // namespace ml


#endif   // __ml_kmeans_h__
