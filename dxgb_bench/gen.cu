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
template <typename Fn>
int Dispatch(bool is_cuda, Fn fn) {
  if (is_cuda) {
    fn(thrust::cuda::par_nosync);
    return static_cast<int>(cub::SyncStream(cudaStreamPerThread));
  }
  omp_set_num_threads(std::thread::hardware_concurrency());
  fn(thrust::omp::par);
  return 0;
}

template <typename Exec>
void DenseRegression(Exec exec, int64_t n_samples, int64_t n_columns, int64_t n_targets,
                     double sparsity, int64_t seed, float *out, float *y) {
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
  return Dispatch(
      is_cuda, [&](auto exec) { DenseRegression(exec, m, n, n_targets, sparsity, seed, out, y); });
}

template <typename Exec>
void ImbalancedFeatures(Exec exec, int64_t m, int64_t n, int64_t n_binary, int64_t seed,
                        int64_t row_offset, float *out) {
  thrust::for_each_n(exec, thrust::make_counting_iterator(int64_t{0}), m * n,
                     [=] __host__ __device__(int64_t i) {
                       thrust::minstd_rand rng(seed);
                       // Reserve two draws per cell for the normal distribution.
                       rng.discard(2 * (row_offset * n + i));
                       if (i % n < n_binary) {
                         thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
                         out[i] = dist(rng) >= 0.5f ? 1.0f : 0.0f;
                       } else {
                         thrust::normal_distribution<float> dist(0.0f, 1.0f);
                         out[i] = dist(rng);
                       }
                     });
}

int MakeImbalancedFeatures(bool is_cuda, int64_t m, int64_t n, int64_t n_binary, int64_t seed,
                           int64_t row_offset, float *out) {
  return Dispatch(
      is_cuda, [&](auto exec) { ImbalancedFeatures(exec, m, n, n_binary, seed, row_offset, out); });
}

template <typename Exec>
void RegressionTargets(Exec exec, int64_t m, int64_t n, int64_t n_targets, int64_t seed,
                       int64_t row_offset, float const *X, float const *coef, float *y) {
  thrust::for_each_n(exec, thrust::make_counting_iterator(int64_t{0}), m * n_targets,
                     [=] __host__ __device__(int64_t i) {
                       auto row = i / n_targets;
                       auto target = i % n_targets;
                       // Keep observation noise separate from the feature stream.
                       thrust::minstd_rand rng(seed ^ 0x5bd1e995);
                       rng.discard(2 * (row_offset * n_targets + i));
                       thrust::normal_distribution<float> dist(0.0f, 1.0f);
                       float value = dist(rng);
                       for (int64_t j = 0; j < n; ++j) {
                         value += X[row * n + j] * coef[j * n_targets + target];
                       }
                       y[i] = value;
                     });
}

int MakeRegressionTargets(bool is_cuda, int64_t m, int64_t n, int64_t n_targets, int64_t seed,
                          int64_t row_offset, float const *X, float const *coef, float *y) {
  return Dispatch(is_cuda, [&](auto exec) {
    RegressionTargets(exec, m, n, n_targets, seed, row_offset, X, coef, y);
  });
}
}  // namespace cuda_impl
