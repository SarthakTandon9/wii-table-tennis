#pragma once
// The static world (table, net, floor), paddles, and one simulation step.
//
// Collision strategy, chosen per obstacle by how fast things move relative to
// how thick they are:
//  - Table and floor: the ball crosses a horizontal plane. We find the time of
//    impact inside the step by interpolation, bounce there, and spend the rest
//    of the step moving away. No tunnelling at any speed.
//  - Paddle: both ball and paddle move fast, and the paddle is thin. This is a
//    swept test done in the paddle's own frame, where the paddle is a
//    stationary disc and the ball moves with the relative velocity.
//  - Net: a thin rectangle, but the ball is 40 mm wide and moves under 40 mm
//    per 1 ms step, so a closest-point overlap test cannot miss it.

#include "core/types.hpp"
#include "math/vec3.hpp"
#include "physics/ball.hpp"

#include <cstdint>
#include <span>

namespace pp {

struct Table {
  real length = 2.74f;   // along z
  real width = 1.525f;   // along x
  real height = 0.76f;   // playing surface y
  // e = 0.87: the ITTF test (a 30 cm drop must rebound about 23 cm) once drag
  // is taken out.
  SurfaceParams surface{.restitution = 0.87f, .friction = 0.25f};
};

struct Net {
  real height = 0.1525f;    // above the table surface
  real overhang = 0.1525f;  // past each side of the table
  SurfaceParams surface{.restitution = 0.15f, .friction = 0.30f};
};

struct Floor {
  SurfaceParams surface{.restitution = 0.75f, .friction = 0.40f};
};

// A paddle is a disc. `pos` is its centre at the START of the step; it moves
// with constant `vel` during the step. `normal` is a unit vector; both faces
// are playable.
struct Paddle {
  Vec3 pos;
  Vec3 vel;
  Vec3 normal{0, 0, -1};
  real radius = 0.09f;
  SurfaceParams surface{.restitution = 0.80f, .friction = 0.50f};
  bool enabled = true;  // false: the ball passes through (used right after a hit)
};

// What the ball touched during a step. Bit flags so one step can report
// several (for example paddle then table).
using ContactMask = std::uint8_t;
namespace contact {
inline constexpr ContactMask kNone = 0;
inline constexpr ContactMask kTableNear = 1 << 0;  // player's half (z > 0)
inline constexpr ContactMask kTableFar = 1 << 1;   // opponent's half (z < 0)
inline constexpr ContactMask kNet = 1 << 2;
inline constexpr ContactMask kFloor = 1 << 3;
inline constexpr ContactMask kPaddle0 = 1 << 4;  // paddles[0]
inline constexpr ContactMask kPaddle1 = 1 << 5;  // paddles[1]
}  // namespace contact

struct World {
  BallState ball;
  BallParams ball_params;
  Table table;
  Net net;
  Floor floor;
};

// Advance the ball by dt, resolving collisions. At most two paddles.
// Deterministic: same inputs, same bits out.
ContactMask step(World& world, std::span<const Paddle> paddles, real dt);

}  // namespace pp
