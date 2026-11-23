#pragma once

#include <cstddef>

namespace pp {

// How far apart two variables must be so that two cores writing to them do
// not fight over the same cache line ("false sharing").
//
// This machine gives three different answers:
//   sysctl hw.cachelinesize                          128
//   std::hardware_destructive_interference_size      256   (Apple clang)
//   the "everyone knows" x86 figure                   64
// 128 is the real line size of Apple silicon. bench/queue_bench measures all
// of them; see the numbers there before changing this.
inline constexpr std::size_t kCacheLine = 128;

}  // namespace pp
