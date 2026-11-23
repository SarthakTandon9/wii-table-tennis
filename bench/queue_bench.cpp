// How much does the lock-free queue buy, and how much of that is layout?
//
// One thread pushes N integers, another pops them. Reported: nanoseconds per
// item as best / median / worst over several runs, in two regimes:
//
//   flat out   both threads do nothing but push and pop
//   with work  each thread does a little unrelated computation per item,
//              which is what real producers and consumers look like
//
//   mutex ring        the same ring buffer guarded by std::mutex
//   spsc, align 8     lock-free, producer and consumer indices side by side
//                     on one cache line (false sharing)
//   spsc, align 64    separated by the x86 cache-line size
//   spsc, align 128   separated by the real Apple-silicon line size
//   spsc, align 256   what std::hardware_destructive_interference_size says
//
// Build with the release preset; a debug build measures nothing useful.
//
// READ THE SPREAD, NOT JUST THE BEST. What this showed on an M5 (2026-11):
//  - The mutex ring is steady at about 22-25 ns per item flat out.
//  - The unpadded lock-free ring is steady at about 8-12 ns.
//  - The padded rings are BIMODAL: about 8 ns on some runs and 25-39 ns on
//    others, from the very same binary. Which one you get is decided by
//    where the scheduler puts the two threads (macOS has no affinity API, and
//    the machine has two kinds of core), not by the code.
// So this benchmark cannot, on its own, settle 64 vs 128 vs 256, and a single
// run of it can "prove" either that padding helps or that it hurts. Settling
// it needs per-core cache-miss counters (Instruments, M6). Until then the
// ring keeps 128, the real line size, on first principles.
#include "concurrency/qos.hpp"
#include "concurrency/spsc_ring.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

namespace {

constexpr std::size_t kCapacity = 1024;
constexpr std::uint64_t kItems = 20'000'000;
constexpr int kRuns = 7;

template <typename T, std::size_t Capacity>
class MutexRing {
 public:
  bool try_push(const T& value) {
    std::lock_guard lock{mutex_};
    if (tail_ - head_ == Capacity) return false;
    slots_[tail_++ & (Capacity - 1)] = value;
    return true;
  }
  bool try_pop(T& out) {
    std::lock_guard lock{mutex_};
    if (head_ == tail_) return false;
    out = slots_[head_++ & (Capacity - 1)];
    return true;
  }

 private:
  std::mutex mutex_;
  std::size_t head_ = 0, tail_ = 0;
  std::array<T, Capacity> slots_;
};

// The polite way to spin: a hint that this core is waiting on another one.
// Without it a spin loop hammers the cache line it is polling and slows down
// the very thread it is waiting for.
inline void cpu_relax() {
#if defined(__aarch64__)
  __asm__ __volatile__("isb");  // what Apple silicon spin loops use as a pause
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

// Stand-in for the real work a thread does between queue operations.
inline void unrelated_work(int amount) {
  volatile std::uint64_t x = 0;
  for (int i = 0; i < amount; ++i) x = x + static_cast<std::uint64_t>(i);
}

// Busy-waits (politely) when the queue is full or empty: this measures the
// queue, not the scheduler.
template <typename Queue>
double ns_per_item(Queue& queue, int work) {
  const auto start = std::chrono::steady_clock::now();
  std::thread producer([&] {
    pp::make_current_thread_interactive();
    for (std::uint64_t i = 1; i <= kItems; ++i) {
      unrelated_work(work);
      while (!queue.try_push(i)) cpu_relax();
    }
  });

  pp::make_current_thread_interactive();
  std::uint64_t value = 0, sum = 0;
  for (std::uint64_t received = 0; received < kItems;) {
    if (queue.try_pop(value)) {
      sum += value;
      ++received;
      unrelated_work(work);
    } else {
      cpu_relax();
    }
  }
  producer.join();
  const auto end = std::chrono::steady_clock::now();

  if (sum != kItems * (kItems + 1) / 2) std::printf("  !! checksum mismatch\n");
  return std::chrono::duration<double, std::nano>(end - start).count() /
         static_cast<double>(kItems);
}

struct Spread {
  double best, median, worst;
};

template <typename Queue>
Spread measure(int work) {
  std::array<double, kRuns> ns{};
  for (double& run : ns) {
    // Heap-allocated so every run, and every variant, starts with a fresh
    // queue at its own alignment.
    auto queue = std::make_unique<Queue>();
    run = ns_per_item(*queue, work);
  }
  std::sort(ns.begin(), ns.end());
  return {ns.front(), ns[kRuns / 2], ns.back()};
}

template <typename Queue>
void report(const char* name) {
  const Spread flat = measure<Queue>(0);
  const Spread busy = measure<Queue>(20);
  std::printf("  %-16s %6.1f /%6.1f /%6.1f     %6.1f /%6.1f /%6.1f\n", name, flat.best,
              flat.median, flat.worst, busy.best, busy.median, busy.worst);
}

}  // namespace

int main() {
  std::printf("%llu items through a %zu-slot queue, %d runs each, ns per item\n\n",
              static_cast<unsigned long long>(kItems), kCapacity, kRuns);
  std::printf("  %-16s %-26s %s\n", "", "flat out", "with work");
  std::printf("  %-16s %-26s %s\n", "", "best / median / worst", "best / median / worst");
  report<MutexRing<std::uint64_t, kCapacity>>("mutex ring");
  report<pp::SpscRing<std::uint64_t, kCapacity, 8>>("spsc, align 8");
  report<pp::SpscRing<std::uint64_t, kCapacity, 64>>("spsc, align 64");
  report<pp::SpscRing<std::uint64_t, kCapacity, 128>>("spsc, align 128");
  report<pp::SpscRing<std::uint64_t, kCapacity, 256>>("spsc, align 256");
  return 0;
}
