// How fast can candidate shots be evaluated, and what does each technique buy?
//
// Part 1: one thread, four implementations of the same simulation.
// Part 2: the fastest one, spread over more and more threads.
//
// Build with the release preset. Timings are the median of several runs,
// with the fastest and slowest alongside so the spread is visible.
//
// Measured on an M5 (4 performance + 6 efficiency cores), 2026-11, 16384
// shots of about 194 steps each:
//
//   array of structs, scalar          5700 ns/shot   1.0x
//   SoA, compiler left alone          8000 ns/shot   0.7x   (it did not vectorise)
//   SoA, vectorisation forced         2500 ns/shot   2.3x
//   SoA, forced, in cache blocks      1900 ns/shot   3.1x
//   NEON, lanes idle until all land   2400 ns/shot   2.5x   (first hand-written try)
//   NEON, lanes refilled              1500 ns/shot   3.8x
//
//   NEON refilled on 4 threads         470 ns/shot   3.3x of one thread
//   NEON refilled on 10 threads        230 ns/shot   6.5x of one thread
//
// What each step taught:
//  - Changing the layout alone made it slower. Vector-friendly data is not
//    vector code: the compiler's cost model declined the loop.
//  - Forcing vectorisation gave 2.3x, not 4x, because every lane runs until
//    the slowest ball in the batch lands. Blocking shortens that wait (the
//    slowest of 64, not of 16384) and keeps the state in L1.
//  - Hand-written code only beat the compiler once it did something the
//    compiler cannot: refill a finished lane with new work.
//  - Ten cores give 6.5x, not 10x: six of them are efficiency cores.
#include "concurrency/qos.hpp"
#include "concurrency/thread_pool.hpp"
#include "planner/shot_batch.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace pp;

namespace {

constexpr std::size_t kCandidates = 16384;
constexpr int kRuns = 9;

std::vector<ShotCandidate> make_candidates(std::size_t n) {
  std::vector<ShotCandidate> out(n);
  std::uint32_t state = 0xFEEDu;
  auto unit = [&] {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 16777216.0f;
  };
  for (ShotCandidate& c : out) {
    c.vel = {-2.0f + 4.0f * unit(), -1.0f + 4.5f * unit(), -(3.0f + 9.0f * unit())};
    c.spin = {-200.0f + 400.0f * unit(), -60.0f + 120.0f * unit(), -40.0f + 80.0f * unit()};
  }
  return out;
}

struct Timing {
  double best, median, worst;  // microseconds for the whole batch
};

template <typename F>
Timing time_it(F&& run) {
  std::array<double, kRuns> us{};
  run();  // warm-up: page in memory, wake the threads
  for (double& t : us) {
    const auto start = std::chrono::steady_clock::now();
    run();
    t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
  }
  std::sort(us.begin(), us.end());
  return {us.front(), us[kRuns / 2], us.back()};
}

void print_row(const char* name, Timing t, double baseline_us) {
  std::printf("  %-22s %8.0f us  (%6.0f - %6.0f)   %6.1f ns/shot   %5.1fx\n", name, t.median,
              t.best, t.worst, 1000.0 * t.median / static_cast<double>(kCandidates),
              baseline_us / t.median);
}

}  // namespace

int main() {
  make_current_thread_interactive();
  ShotQuery q;
  q.from = {0.2f, 0.95f, 1.45f};
  const auto candidates = make_candidates(kCandidates);

  std::vector<ShotOutcome> outcomes(kCandidates);
  ShotBatch batch;
  batch.resize(kCandidates);
  for (std::size_t i = 0; i < kCandidates; ++i) batch.set(i, candidates[i]);
  const std::size_t n = batch.padded();

  // How much work is in there? Average flight length in steps, from the
  // reference results.
  simulate_aos(q, candidates, outcomes);
  double steps = 0;
  for (const ShotOutcome& o : outcomes) {
    steps += o.time >= 0 ? static_cast<double>(o.time / q.dt) : static_cast<double>(q.max_steps);
  }
  std::printf("%zu candidate shots, %.0f integration steps each on average\n\n", kCandidates,
              steps / static_cast<double>(kCandidates));

  std::printf("One thread                 median     (best - worst)      per shot    speed-up\n");
  const Timing aos = time_it([&] { simulate_aos(q, candidates, outcomes); });
  print_row("array of structs", aos, aos.median);
  print_row("SoA, whole batch", time_it([&] { simulate_soa_wide(q, batch, 0, n); }), aos.median);
  print_row("SoA, cache blocks", time_it([&] { simulate_soa_blocked(q, batch, 0, n); }),
            aos.median);
  const Timing neon = time_it([&] { simulate_neon(q, batch, 0, n); });
  print_row("NEON, hand-written", neon, aos.median);

  std::printf("\nNEON across threads        median     (best - worst)      per shot    vs 1 thread\n");
  for (const unsigned threads : {1u, 2u, 3u, 4u, 6u, 8u, 10u}) {
    ThreadPool pool{threads - 1};
    const Timing t = time_it([&] {
      pool.parallel_for(n, 256, [&](std::size_t begin, std::size_t end) {
        simulate_neon(q, batch, begin, end);
      });
    });
    char name[32];
    std::snprintf(name, sizeof name, "%u thread%s", threads, threads == 1 ? "" : "s");
    print_row(name, t, neon.median);
  }
  return 0;
}
