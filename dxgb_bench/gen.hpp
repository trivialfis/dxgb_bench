#include <cstdint>

namespace cuda_impl {
int MakeDenseRegression(int64_t m, int64_t n, double sparsity, int64_t seed,
                        float *out, float *y);
}
