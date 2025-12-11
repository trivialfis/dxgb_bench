#include <omp.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/system/omp/execution_policy.h>  // for par

#include <cmath>  // for isnan
#include <limits>
#include <thread>  // for hardware_concurrency

namespace cuda_impl {
template <typename Exec>
void Impl(Exec exec, int64_t n_samples, int64_t n_columns, int64_t n_targets, double sparsity,
          int64_t seed, float *out, float *y) {
  thrust::for_each_n(exec, thrust::make_counting_iterator(0ul), n_samples * n_columns,
                     [=] __host__ __device__(std::size_t i) {
                       thrust::minstd_rand rng, rng1;
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

  // We can run this as a single kernel if we have an unravel impl.
  for (int64_t t = 0; t < n_targets; ++t) {
    thrust::minstd_rand rng;
    rng.seed(0);
    rng.discard(t);
    thrust::normal_distribution<float> dist{0.0f, 1.5f};
    auto err = dist(rng);

    thrust::for_each_n(exec, thrust::make_counting_iterator(0ul), n_samples,
                       [=] __host__ __device__(std::size_t i) {
                         y[i * n_targets + t] = 0;

                         for (std::size_t j = 0; j < n_columns; ++j) {
                           if (!isnan(out[n_columns * i + j])) {
                             y[i * n_targets + t] += out[n_columns * i + j] * err;
                           }
                         }
                       });
  }
}

int MakeDenseRegression(bool is_cuda, int64_t m, int64_t n, int64_t n_targets, double sparsity,
                        int64_t seed, float *out, float *y) {
  if (is_cuda) {
    Impl(thrust::cuda::par_nosync, m, n, n_targets, sparsity, seed, out, y);
    cub::SyncStream(cudaStreamPerThread);
  } else {
    omp_set_num_threads(std::thread::hardware_concurrency());
    Impl(thrust::omp::par, m, n, n_targets, sparsity, seed, out, y);
  }
  return 0;
}
}  // namespace cuda_impl
