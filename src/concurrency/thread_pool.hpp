#pragma once
// A small pool of worker threads for splitting one loop across cores.
//
// parallel_for(count, chunk, f) calls f(begin, end) over [0, count) in pieces
// of `chunk`, on the workers and on the calling thread, and returns when all
// pieces are done.
//
// How the work is shared out: there is no queue of tasks. There is one shared
// counter, "the next piece nobody has taken". Each thread repeatedly claims a
// piece with a single atomic fetch_add and runs it. This is dynamic
// scheduling: a thread on a fast core simply claims more pieces than one on a
// slow core, which matters on a chip with two kinds of core. Splitting the
// range evenly up front would leave the fast cores idle, waiting for the slow
// ones to finish their share.
//
// Threads are created once and sleep between calls. Creating a thread costs
// tens of microseconds; a planning call may only take a few hundred.

#include "concurrency/cache_line.hpp"
#include "concurrency/qos.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace pp {

class ThreadPool {
 public:
  // `workers` extra threads; the caller of parallel_for is one more.
  explicit ThreadPool(unsigned workers) {
    threads_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) {
      threads_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard lock{mutex_};
      stopping_ = true;
    }
    wake_.notify_all();
  }  // jthreads join here

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned threads() const { return static_cast<unsigned>(threads_.size()) + 1; }

  // Not re-entrant: one parallel_for at a time, from one thread.
  template <typename F>
  void parallel_for(std::size_t count, std::size_t chunk, F&& body) {
    if (count == 0) return;
    // The workers run the body through a plain function pointer plus a
    // pointer to the caller's lambda. That avoids std::function, which may
    // allocate, and is safe because this call does not return until every
    // worker has finished with it.
    job_.run = [](void* context, std::size_t begin, std::size_t end) {
      (*static_cast<std::remove_reference_t<F>*>(context))(begin, end);
    };
    job_.context = &body;
    job_.count = count;
    job_.chunk = std::max<std::size_t>(chunk, 1);
    next_.value.store(0, std::memory_order_relaxed);
    {
      std::lock_guard lock{mutex_};
      ++generation_;
      busy_ = static_cast<unsigned>(threads_.size());
    }
    wake_.notify_all();

    run_pieces();  // the caller works too

    std::unique_lock lock{mutex_};
    done_.wait(lock, [this] { return busy_ == 0; });
  }

 private:
  struct Job {
    void (*run)(void*, std::size_t, std::size_t) = nullptr;
    void* context = nullptr;
    std::size_t count = 0, chunk = 1;
  };

  void run_pieces() {
    for (;;) {
      const std::size_t begin = next_.value.fetch_add(job_.chunk, std::memory_order_relaxed);
      if (begin >= job_.count) return;
      job_.run(job_.context, begin, std::min(begin + job_.chunk, job_.count));
    }
  }

  void worker_loop() {
    make_current_thread_interactive();
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock lock{mutex_};
        wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
      }
      run_pieces();
      {
        std::lock_guard lock{mutex_};
        --busy_;
      }
      done_.notify_one();
    }
  }

  // Every thread hammers this one counter, so it gets a cache line to itself.
  struct alignas(kCacheLine) Counter {
    std::atomic<std::size_t> value{0};
  };

  Job job_;
  Counter next_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  std::uint64_t generation_ = 0;
  unsigned busy_ = 0;
  bool stopping_ = false;
  std::vector<std::jthread> threads_;  // last: started after, joined before, the rest
};

}  // namespace pp
