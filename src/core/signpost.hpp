#pragma once
// Named time intervals that show up in Instruments.
//
// A sampling profiler says which functions are hot. It cannot say "this
// particular wake-up took 3 ms" or "Vision was slow on that frame". Signposts
// can: each marks the start and end of a named stretch of work, and
// Instruments (the Points of Interest / os_signpost instruments) draws them
// on a timeline per thread, next to the samples.
//
//   {
//     PP_INTERVAL("predict");   // open until the end of this scope
//     ...
//   }
//
// When nothing is recording, a signpost costs a few nanoseconds (one check of
// an enabled flag), so they stay in release builds.

#if defined(__APPLE__)
#include <os/log.h>
#include <os/signpost.h>

namespace pp {

inline os_log_t signpost_log() {
  // The "PointsOfInterest" category is the one Instruments shows by default.
  static const os_log_t log = os_log_create("local.palmrally", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
  return log;
}

// One interval, tied to a scope. The name must be a string literal: the
// signpost API stores the pointer, not a copy.
#define PP_INTERVAL_IMPL(name, line)                                                       \
  const os_signpost_id_t pp_signpost_id_##line = os_signpost_id_generate(::pp::signpost_log()); \
  os_signpost_interval_begin(::pp::signpost_log(), pp_signpost_id_##line, name);          \
  const ::pp::SignpostEnd pp_signpost_end_##line{[&] {                                     \
    os_signpost_interval_end(::pp::signpost_log(), pp_signpost_id_##line, name);          \
  }}

// Runs a callable when it goes out of scope.
template <typename F>
struct SignpostEnd {
  F at_exit;
  ~SignpostEnd() { at_exit(); }
};
template <typename F>
SignpostEnd(F) -> SignpostEnd<F>;

}  // namespace pp

#define PP_INTERVAL_EXPAND(name, line) PP_INTERVAL_IMPL(name, line)
#define PP_INTERVAL(name) PP_INTERVAL_EXPAND(name, __LINE__)

#else
#define PP_INTERVAL(name) ((void)0)
#endif
