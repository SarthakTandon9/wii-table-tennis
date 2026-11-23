#pragma once
// Simulating many candidate shots at once.
//
// The question a planner asks is "if the ball leaves here with this velocity
// and spin, where does it land and does it clear the net?", for thousands of
// velocities. Each answer is a few hundred integration steps. The work is the
// same arithmetic repeated across independent candidates, which is exactly
// the shape SIMD hardware is built for.
//
// Four implementations of the same function, kept side by side so their cost
// can be compared (bench/planner_bench):
//
//   simulate_aos          one candidate at a time, array-of-structs. The
//                         straightforward version and the reference answer.
//   simulate_soa_wide     struct-of-arrays; each time step sweeps the whole
//                         batch. The compiler vectorises the inner loop, but
//                         only when told to (see the pragma in the .cpp).
//   simulate_soa_blocked  the same loop, cut into blocks small enough to stay
//                         in the L1 cache for all their time steps.
//   simulate_neon         hand-written NEON, four candidates per register,
//                         state held in registers, and each lane refilled
//                         with the next candidate as soon as its ball lands.
//
// Why the data layout matters: with an array of structs, the x of candidate
// 0 sits next to the y of candidate 0. A vector instruction wants four x's
// side by side. Struct-of-arrays stores them that way, so one load fetches
// four candidates' worth of the same quantity.

#include "core/types.hpp"
#include "math/vec3.hpp"
#include "physics/ball.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pp {

// What all candidates in a batch share.
struct ShotQuery {
  Vec3 from;                 // where the ball is struck
  BallParams ball;
  real land_y = 0.78f;       // ball-centre height that counts as "on the table"
  real net_top = 0.9125f;    // height of the top of the net
  real dt = 2e-3f;
  int max_steps = 700;       // 1.4 s of flight
};

// --- array-of-structs form ---
struct ShotCandidate {
  Vec3 vel, spin;
};
struct ShotOutcome {
  real land_x = 0, land_z = 0;
  real time = -1;            // seconds to landing; -1 if it never came down
  real clearance = -100;     // gap over the net where it crossed; -100 if it never crossed
};

// --- struct-of-arrays form ---
// Inputs and outputs as parallel arrays. Sizes are rounded up to a multiple
// of 4 so vector code never needs a special case for the tail; the padding
// candidates are harmless (a ball dropped from rest).
struct ShotBatch {
  std::vector<float> vx, vy, vz, wx, wy, wz;         // in
  std::vector<float> land_x, land_z, time, clearance;  // out
  std::size_t count = 0;  // real candidates, excluding padding

  void resize(std::size_t n);
  std::size_t padded() const { return vx.size(); }
  void set(std::size_t i, const ShotCandidate& c);
  ShotOutcome outcome(std::size_t i) const { return {land_x[i], land_z[i], time[i], clearance[i]}; }
};

void simulate_aos(const ShotQuery& q, std::span<const ShotCandidate> in, std::span<ShotOutcome> out);

// Each of these may be called on a sub-range [begin, end) of the batch, both
// multiples of 4, so that several threads can share one batch.
void simulate_soa_wide(const ShotQuery& q, ShotBatch& batch, std::size_t begin, std::size_t end);
void simulate_soa_blocked(const ShotQuery& q, ShotBatch& batch, std::size_t begin, std::size_t end);
void simulate_neon(const ShotQuery& q, ShotBatch& batch, std::size_t begin, std::size_t end);

}  // namespace pp
