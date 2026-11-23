#pragma once
// "Latest value wins" handoff from one writer thread to one reader thread.
//
// A queue is the wrong tool for sending the simulation state to the renderer:
// the renderer does not want every state, it wants the newest one, and it
// must never make the simulation wait. A triple buffer gives exactly that,
// wait-free on both sides.
//
// Three slots. At any moment:
//   - the writer owns one (it is filling it in),
//   - the reader owns one (it is drawing from it),
//   - the third sits in the middle holding the most recently published value.
// Publishing swaps the writer's slot with the middle one; reading swaps the
// reader's slot with the middle one. Each swap is a single atomic exchange,
// so neither side can ever see a half-written value and neither ever waits.
//
// The middle slot's index and a "there is something new" flag share one
// atomic byte, which is what makes the swap a single instruction.

#include "concurrency/cache_line.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>

namespace pp {

template <typename T>
  requires std::is_trivially_copyable_v<T>
class TripleBuffer {
 public:
  // Writer thread only.
  void publish(const T& value) {
    slots_[writer_.index].value = value;
    // Hand over our slot, marked fresh; take whatever was in the middle.
    const std::uint8_t previous =
        middle_.exchange(static_cast<std::uint8_t>(writer_.index | kFresh),
                         std::memory_order_acq_rel);
    writer_.index = previous & kIndexMask;
  }

  // Reader thread only. Returns the newest published value; the reference
  // stays valid (and unchanged) until the next call to latest().
  const T& latest() {
    // Only swap when there is something new. Otherwise we would trade our
    // current value for the stale one that the writer handed back earlier.
    if (middle_.load(std::memory_order_acquire) & kFresh) {
      const std::uint8_t previous = middle_.exchange(reader_.index, std::memory_order_acq_rel);
      reader_.index = previous & kIndexMask;
    }
    return slots_[reader_.index].value;
  }

 private:
  static constexpr std::uint8_t kFresh = 0x4;
  static constexpr std::uint8_t kIndexMask = 0x3;

  // Each slot and each side's private index gets its own cache line, so the
  // writer filling in its slot does not disturb the reader's.
  struct alignas(kCacheLine) Slot {
    T value{};
  };
  struct alignas(kCacheLine) Owned {
    std::uint8_t index;
  };

  std::array<Slot, 3> slots_{};
  Owned writer_{0};
  Owned reader_{1};
  alignas(kCacheLine) std::atomic<std::uint8_t> middle_{2};
};

}  // namespace pp
