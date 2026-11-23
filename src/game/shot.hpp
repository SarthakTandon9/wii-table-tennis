#pragma once
// Shot planning by simulation.
//
// There is no closed form for "what velocity lands the ball there" once drag
// and spin are in the model, so every question here is answered by running
// the same physics the game runs and correcting the guess (a shooting
// method). This is the seed of the M6 batch planner: today it evaluates a
// handful of candidates serially; later it evaluates thousands in SIMD.

#include "core/types.hpp"
#include "math/vec3.hpp"
#include "physics/world.hpp"

#include <optional>

namespace pp {

// Result of flying a ball (no collisions) until it comes down to table height.
struct Flight {
  Vec3 landing;            // ball centre when it reaches table height
  real time = 0;           // seconds until then
  real net_clearance = 0;  // gap between ball and net top where it crossed z = 0
  bool crossed_net = false;
  bool landed = false;
};

Flight fly(const World& world, BallState start, real max_time = 3.0f);

// Velocity that takes the ball from `from` to land at `target` (x, z) after
// roughly `flight_time` seconds, with drag and spin accounted for.
std::optional<Vec3> aim(const World& world, Vec3 from, Vec3 spin, Vec3 target, real flight_time);

// Does a rally stroke starting like this bounce once on the other half and
// then carry past the end of the table (so that it can be played back)?
bool stroke_carries(World world, const BallState& start);

// A stroke across the net: lands at `target`, travelling at about `speed`
// horizontally, lofted just enough to clear the net.
std::optional<Vec3> plan_shot(const World& world, Vec3 from, Vec3 spin, Vec3 target, real speed);

// A legal serve from `from`: bounces once on the server's own half, clears
// the net without touching it, lands on the other half and carries past the
// end line. `aim_x` steers it sideways, `pace` in [0, 1] picks between a slow
// looping serve and a fast flat one. The planner chooses the topspin too: a
// ball struck low has to be lifted, and without topspin it loses too much
// speed at the first bounce to reach the far end.
struct ServePlan {
  Vec3 vel;
  Vec3 spin;
};
std::optional<ServePlan> plan_serve(const World& world, Vec3 from, real aim_x, real pace);

// Where will the ball pass through a plane?
struct Crossing {
  Vec3 point;
  real time = 0;
  bool bounced = false;  // touched the table on the way (so it is legal to hit)
};

// Full simulation (table, net, floor) of the world's current ball against the
// plane through `plane_point` whose unit normal `toward_ball` points at the
// side the ball arrives from. Pass `already_bounced` when the ball has had its
// bounce on the receiving half before this call. A ball that would bounce
// there twice cannot be returned, so that case reports no crossing.
std::optional<Crossing> predict_crossing(const World& world, Vec3 plane_point, Vec3 toward_ball,
                                         bool already_bounced = false, real horizon = 1.6f);

}  // namespace pp
