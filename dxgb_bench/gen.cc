#include <random>
#include <thread>

extern "C" {
__attribute__((visibility("default"))) int
MakeDenseRegression(int64_t m, int64_t n, int64_t seed, float *out, float *y) {
  auto n_threads = std::thread::hardware_concurrency();
  auto n_samples_per_threads = static_cast<double>(m) / n_threads;

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < n_threads; ++i) {
    workers.emplace_back([=] {
      std::size_t n_samples = n_samples_per_threads;
      if (i == n_threads - 1) {
        auto prev = n_samples_per_threads * (n_threads - 1);
        n_samples = m - prev;
      }
      auto begin = n_samples_per_threads * i;
      std::default_random_engine rng;
      rng.seed(i + seed * m);
      std::normal_distribution<float> dist{0.0f, 1.5f};
      for (std::size_t j = begin; j < begin + n_samples; ++j) {
        for (std::size_t k = 0; k < n; ++k) {
          auto v = dist(rng);
          auto idx = j * n + k;
          out[idx] = v;
        }
      }

      for (std::size_t j = begin; j < begin + n_samples; ++j) {
        y[j] = 0.0f;
        for (std::size_t k = 0; k < n; ++k) {
          auto err = dist(rng);
          auto idx = j * n + k;
          y[j] += (out[idx] + err);
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
