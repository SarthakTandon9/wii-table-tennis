#pragma once
// Single-producer, single-consumer bounded queue. Wait-free: push and pop
// each finish in a fixed number of steps and never block, lock or allocate.
//
// The contract that makes it this simple: exactly ONE thread ever calls
// try_push and exactly ONE (other) thread ever calls try_pop. Under that
// contract each index has a single writer, so no compare-and-swap is needed;
// a plain store is enough.
//
// How the two threads agree, and why each memory order is what it is:
//
//   tail_  written only by the producer, read by the consumer
//   head_  written only by the consumer, read by the producer
//
//   producer:  slot[tail] = value;  tail_.store(tail + 1, release)
//   consumer:  t = tail_.load(acquire);  ... read slot[head] ...
//
// The release store / acquire load pair is the whole synchronisation. It
// guarantees that when the consumer sees the new tail, it also sees the write
// to the slot that happened before it. Without it the consumer could observe
// the index move before the data arrived. The mirror-image pair on head_
// protects the producer from overwriting a slot the consumer is still reading.
//
// A thread loads its OWN index with `relaxed`: nobody else writes it, so
// there is nothing to synchronise with.
//
// Layout: the producer's variables and the consumer's variables are kept on
// separate cache lines. Otherwise every push would invalidate the line the
// consumer is reading and every pop the producer's, and the two cores would
// spend their time passing one line back and forth. The `Align` parameter
// exists so the benchmark can measure exactly that.
//
// Each side also keeps a private, possibly stale copy of the other side's
// index. The producer only needs to know "is there room"; if its old copy of
// head already says yes, it need not touch the consumer's line at all.

#include "concurrency/cache_line.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <type_traits>

namespace pp {

template <typename T, std::size_t Capacity, std::size_t Align = kCacheLine>
  requires std::is_trivially_copyable_v<T> && (std::has_single_bit(Capacity))
class SpscRing {
 public:
  // Producer thread only. Returns false if the queue is full.
  bool try_push(const T& value) {
    const std::size_t tail = producer_.tail.load(std::memory_order_relaxed);
    if (tail - producer_.head_cache == Capacity) {
      producer_.head_cache = consumer_.head.load(std::memory_order_acquire);
      if (tail - producer_.head_cache == Capacity) return false;
    }
    slots_[tail & kMask] = value;
    producer_.tail.store(tail + 1, std::memory_order_release);
    return true;
  }

  // Consumer thread only. Returns false if the queue is empty.
  bool try_pop(T& out) {
    const std::size_t head = consumer_.head.load(std::memory_order_relaxed);
    if (head == consumer_.tail_cache) {
      consumer_.tail_cache = producer_.tail.load(std::memory_order_acquire);
      if (head == consumer_.tail_cache) return false;
    }
    out = slots_[head & kMask];
    consumer_.head.store(head + 1, std::memory_order_release);
    return true;
  }

  static constexpr std::size_t capacity() { return Capacity; }

 private:
  // Indices run freely and wrap around size_t; `& kMask` maps them to a slot.
  // That is why Capacity must be a power of two: the mask is then a single
  // AND, and (tail - head) stays correct across the wrap.
  static constexpr std::size_t kMask = Capacity - 1;

  struct alignas(Align) Producer {
    std::atomic<std::size_t> tail{0};
    std::size_t head_cache = 0;
  };
  struct alignas(Align) Consumer {
    std::atomic<std::size_t> head{0};
    std::size_t tail_cache = 0;
  };

  Producer producer_;
  Consumer consumer_;
  alignas(Align) std::array<T, Capacity> slots_;
};

}  // namespace pp
