#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <limits>

namespace cuda_impl {
int MakeDenseRegression(int64_t m, int64_t n, double sparsity, int64_t seed, float *out, float *y) {
  thrust::for_each_n(thrust::cuda::par_nosync, thrust::make_counting_iterator(0ul), m * n,
                     [=] __host__ __device__(std::size_t i) {
                       thrust::default_random_engine rng, rng1;
                       rng.seed(0);
                       rng.discard(i + seed);
                       rng1.seed(0);
                       rng1.discard(i + seed);
                       thrust::normal_distribution<float> dist{0.0f, 1.5f};
                       thrust::uniform_real_distribution<float> miss{0.0f, 1.0f};
                       if (miss(rng1) < sparsity) {
                         out[i] = std::numeric_limits<float>::quiet_NaN();
                         return;
                       }
                       out[i] = dist(rng);
                     });
  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0ul), m,
                     [=] __host__ __device__(std::size_t i) {
                       thrust::default_random_engine rng;
                       rng.seed(0);
                       rng.discard(seed / n + i);
                       thrust::normal_distribution<float> dist{0.0f, 1.5f};
                       auto err = dist(rng);
                       y[i] = err;
                       for (std::size_t j = 0; j < n; ++j) {
                         if (!std::isnan(out[n * i + j])) {
                           y[i] += out[n * i + j];
                         }
                       }
                     });
  return 0;
}
}  // namespace cuda_impl
