#include "gen.hpp"

#include <cassert>  // for assert
#include <cmath>    // for ceil
#include <thread>   // for thread
#include <vector>   // for vector

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

extern "C" {
__attribute__((visibility("default"))) int MakeDenseRegression(bool is_cuda, int64_t m, int64_t n,
                                                               double sparsity, int64_t seed,
                                                               float *out, float *y) {
#if defined(DXGB_USE_CUDA)
  return cuda_impl::MakeDenseRegression(is_cuda, m, n, sparsity, seed, out, y);
#endif
  if (is_cuda) {
    return -1;
  }

  auto n_threads = std::thread::hardware_concurrency();
  auto n_samples_per_threads = std::ceil(static_cast<double>(m) / n_threads);

  std::vector<std::thread> workers;

  for (std::size_t i = 0; i < n_threads; ++i) {
    workers.emplace_back([=] {
      std::size_t n_samples = n_samples_per_threads;  // n_samples for this batch
      if (i == n_threads - 1) {                       // The last batch.
        auto prev = n_samples_per_threads * (n_threads - 1);
        n_samples = m - prev;
      }
      auto begin = n_samples_per_threads * i;
      if (begin >= m) {
        return;
      }
      assert(n_samples <= m);

      std::size_t k = begin * n;
      for (std::size_t j = begin; j < begin + n_samples; ++j) {
        for (std::int64_t fidx = 0; fidx < n; ++fidx) {
          auto idx = j * n + fidx;

          Rng rng, rng1;
          rng.seed(0), rng1.seed(0);
          rng.discard(k + seed), rng1.discard(k + seed);
          Norm dist{0.0f, 1.5f};
          Unif miss{0.0f, 1.0f};

          if (miss(rng1) < sparsity) {
            out[idx] = std::numeric_limits<float>::quiet_NaN();
          } else {
            auto v = dist(rng);
            out[idx] = v;
          }
          k++;
        }
      }

      k = seed / n;  // used with the seed in datagen.
      for (std::size_t j = begin; j < begin + n_samples; ++j) {
        Rng rng;

        rng.seed(0), rng.discard(j + k);
        Norm dist{0.1f, 1.5f};
        auto err = dist(rng);
        y[j] = err;
        for (std::int64_t k = 0; k < n; ++k) {
          auto idx = j * n + k;
          if (!std::isnan(out[idx])) {
            y[j] += out[idx];
          }
        }
      }
    });
  }

  for (auto &t : workers) {
    t.join();
  }

  return 0;
}
}
