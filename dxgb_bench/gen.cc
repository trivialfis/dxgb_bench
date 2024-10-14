#include "gen.hpp"

#include <cassert> // for assert
#include <cmath>   // for ceil
#include <random>  // for default_random_engine
#include <thread>  // for thread

extern "C" {
__attribute__((visibility("default"))) int
MakeDenseRegression(bool is_cuda, int64_t m, int64_t n, double sparsity,
                    int64_t seed, float *out, float *y) {
  if (is_cuda) {
#if defined(DXGB_USE_CUDA)
    return cuda_impl::MakeDenseRegression(m, n, sparsity, seed, out, y);
#else
    return -1;
#endif
  }

  auto n_threads = std::thread::hardware_concurrency();
  auto n_samples_per_threads = std::ceil(static_cast<double>(m) / n_threads);

  std::vector<std::thread> workers;

  for (std::size_t i = 0; i < n_threads; ++i) {
    workers.emplace_back([=] {
      std::size_t n_samples = n_samples_per_threads; // n_samples for this batch
      if (i == n_threads - 1) {                      // The last batch.
        auto prev = n_samples_per_threads * (n_threads - 1);
        n_samples = m - prev;
      }
      auto begin = n_samples_per_threads * i;
      if (begin >= m) {
        return;
      }
      assert(n_samples < m);
      std::default_random_engine rng;
      rng.seed(seed);
      rng.discard(begin);
      std::normal_distribution<float> dist{0.1f, 1.5f};
      std::uniform_real_distribution<float> miss{0.0f, 1.0f};
      for (std::size_t j = begin; j < begin + n_samples; ++j) {
        for (std::int64_t k = 0; k < n; ++k) {
          auto idx = j * n + k;
          if (miss(rng) < sparsity) {
            out[idx] = std::numeric_limits<float>::quiet_NaN();
          } else {
            auto v = dist(rng);
            out[idx] = v;
          }
        }
      }

      for (std::size_t j = begin; j < begin + n_samples; ++j) {
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
