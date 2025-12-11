#include "gen.hpp"

#include <cassert>  // for assert
#include <cmath>    // for ceil

#if defined(DXGB_USE_CUDA)
#include <thrust/random.h>
using Norm = thrust::normal_distribution<float>;
using Unif = thrust::uniform_real_distribution<float>;
using Rng = thrust::minstd_rand;
#else
#include <random>  // for normal_distribution
using Norm = std::normal_distribution<float>;
using Unif = std::uniform_real_distribution<float>;
using Rng = std::minstd_rand;
#endif

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
EXPORT int MakeDenseRegression(bool is_cuda, int64_t m, int64_t n, int64_t n_targets,
                               double sparsity, int64_t seed, float *out, float *y) {
  return cuda_impl::MakeDenseRegression(is_cuda, m, n, n_targets, sparsity, seed, out, y);
}
}
