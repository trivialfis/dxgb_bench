#include <cstdint>

namespace cuda_impl {
int MakeDenseRegression(bool is_cuda, int64_t m, int64_t n, int64_t n_targets, double sparsity,
                        int64_t seed, float *out, float *y);
}
