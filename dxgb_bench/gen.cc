#include "gen.hpp"

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
EXPORT int MakeImbalancedFeatures(bool is_cuda, int64_t m, int64_t n, int64_t n_binary,
                                  int64_t seed, int64_t row_offset, float *out) {
  return cuda_impl::MakeImbalancedFeatures(is_cuda, m, n, n_binary, seed, row_offset, out);
}
EXPORT int MakeRegressionTargets(bool is_cuda, int64_t m, int64_t n, int64_t n_targets,
                                 int64_t seed, int64_t row_offset, float const *X,
                                 float const *coef, float *y) {
  return cuda_impl::MakeRegressionTargets(is_cuda, m, n, n_targets, seed, row_offset, X, coef, y);
}
}
