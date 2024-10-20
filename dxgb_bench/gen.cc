#include "gen.hpp"

#include <cassert>  // for assert
#include <cmath>    // for ceil
#include <random>   // for default_random_engine
#include <thread>   // for thread

extern "C" {
__attribute__((visibility("default"))) int MakeDenseRegression(bool is_cuda, int64_t m, int64_t n,
                                                               double sparsity, int64_t seed,
                                                               float *out, float *y) {
  if (is_cuda) {
#if defined(DXGB_USE_CUDA)
    return cuda_impl::MakeDenseRegression(m, n, sparsity, seed, out, y);
#else
    return -1;
#endif
  }

  auto n_threads = 1;
  auto n_samples_per_threads = 1;

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
      assert(n_samples < m);

      std::uniform_real_distribution<float> miss{0.0f, 1.0f};
      std::size_t k = begin * n;
      for (std::size_t j = begin; j < begin + n_samples; ++j) {
        for (std::int64_t fidx = 0; fidx < n; ++fidx) {
          auto idx = j * n + fidx;

          std::default_random_engine rng, rng1;
          rng.seed(0), rng1.seed(0);
          rng.discard(k + seed), rng1.discard(k + seed);
          std::normal_distribution<float> dist{0.1f, 1.5f};

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
        std::default_random_engine rng;

        rng.seed(0), rng.discard(j + k);
        std::normal_distribution<float> dist{0.1f, 1.5f};
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
