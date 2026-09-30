#include <cstdint>

namespace cuda_impl {
int MakeDenseRegression(bool is_cuda, int64_t m, int64_t n, int64_t n_targets, double sparsity,
                        int64_t seed, float *out, float *y);
int MakeImbalancedFeatures(bool is_cuda, int64_t m, int64_t n, int64_t n_binary, int64_t seed,
                           int64_t row_offset, float *out);
int MakeRegressionTargets(bool is_cuda, int64_t m, int64_t n, int64_t n_targets, int64_t seed,
                          int64_t row_offset, float const *X, float const *coef, float *y);
}
