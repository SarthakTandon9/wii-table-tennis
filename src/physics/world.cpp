#include "physics/world.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace pp {

namespace {

constexpr Vec3 kUp{0, 1, 0};
constexpr real kInfinity = std::numeric_limits<real>::infinity();

// The swept colliders share one contract. Given the ball before (`prev`) and
// after (`ball`) a flight sub-step, they decide whether an impact happened in
// between. If so they rewrite `ball` to the post-bounce state AT THE MOMENT OF
// IMPACT and return the fraction f in [0, 1] of the sub-step that had elapsed.
// The caller then flies the remaining (1 - f) of the sub-step.
//
// Bouncing the end-of-step velocity instead would be simpler, but that
// velocity already includes gravity for time the ball never spent falling.
// Each bounce would then return slightly too much speed: a dropped ball came
// back about 1% too high per bounce when this was written that way.

// Ball crossing a horizontal surface at `height` from above.
std::optional<real> collide_horizontal(BallState& ball, const BallParams& bp,
                                       const BallState& prev, real height, real half_x,
                                       real half_z, const SurfaceParams& surface) {
  const real r = bp.radius;
  if (ball.pos.y - r >= height || ball.vel.y >= 0) return std::nullopt;
  // It must have started above the surface. Otherwise it came in from the
  // side or from underneath (a ball passing below the table).
  if (prev.pos.y - r < height - 1e-4f) return std::nullopt;

  const real drop = prev.pos.y - ball.pos.y;
  const real f = drop > 0 ? std::clamp((prev.pos.y - r - height) / drop, real{0}, real{1})
                          : real{0};
  Vec3 hit = lerp(prev.pos, ball.pos, f);
  if (std::abs(hit.x) > half_x || std::abs(hit.z) > half_z) return std::nullopt;

  hit.y = height + r;
  ball.pos = hit;
  ball.vel = lerp(prev.vel, ball.vel, f);
  bounce(ball, bp, kUp, surface);
  return f;
}

// Swept sphere against a moving disc, solved in the disc's frame. `paddle.pos`
// is the disc centre at the start of this sub-step of length `dt`.
std::optional<real> collide_paddle(BallState& ball, const BallParams& bp, const BallState& prev,
                                   const Paddle& paddle, real dt) {
  const real r = bp.radius;
  const Vec3 n = paddle.normal;

  const Vec3 rel0 = prev.pos - paddle.pos;
  const Vec3 rel1 = ball.pos - (paddle.pos + paddle.vel * dt);

  // Which face is the ball on? Work with distances measured from that face.
  const real d0 = dot(rel0, n);
  const real side = d0 >= 0 ? real{1} : real{-1};
  const real s0 = side * d0;
  const real s1 = side * dot(rel1, n);
  if (s1 >= r) return std::nullopt;  // still clear of the face at the end

  const Vec3 face_normal = side * n;
  if (dot(ball.vel - paddle.vel, face_normal) >= 0) return std::nullopt;  // separating

  // Fraction of the sub-step at which the sphere touches the plane. If it
  // already overlaps (s0 <= r) the contact is "now".
  const real f = s0 > r ? (s0 - r) / (s0 - s1) : real{0};
  const Vec3 rel_hit = lerp(rel0, rel1, f);

  // Is the touch point inside the disc? Half a ball radius of rim tolerance.
  const Vec3 radial = rel_hit - dot(rel_hit, n) * n;
  const real reach = paddle.radius + 0.5f * r;
  if (length_sq(radial) > reach * reach) return std::nullopt;

  // Bounce in the paddle frame (where the paddle is at rest), transform back.
  ball.vel = lerp(prev.vel, ball.vel, f) - paddle.vel;
  bounce(ball, bp, face_normal, paddle.surface);
  ball.vel += paddle.vel;

  // Exactly touching the face, which also removes any initial overlap.
  ball.pos = paddle.pos + paddle.vel * (f * dt) + radial + face_normal * r;
  return f;
}

// Sphere against the net rectangle (in the plane z = 0) by closest point.
// Covers the face, the top tape and the side edges with one formula: the
// contact normal runs from the closest point on the rectangle to the ball.
// Not swept: see the note in world.hpp on why it does not need to be.
bool collide_net(BallState& ball, const BallParams& bp, const BallState& prev,
                 const Table& table, const Net& net) {
  const real r = bp.radius;
  const real half_x = table.width * 0.5f + net.overhang;
  const Vec3 closest{std::clamp(ball.pos.x, -half_x, half_x),
                     std::clamp(ball.pos.y, table.height, table.height + net.height), real{0}};
  const Vec3 offset = ball.pos - closest;
  const real dist_sq = length_sq(offset);
  if (dist_sq >= r * r) return false;

  // Dead centre on the plane: fall back to the side the ball came from.
  const Vec3 n = dist_sq > 1e-12f ? offset / std::sqrt(dist_sq)
                                  : Vec3{0, 0, prev.pos.z >= 0 ? real{1} : real{-1}};
  if (dot(ball.vel, n) >= 0) return false;

  bounce(ball, bp, n, net.surface);
  ball.pos = closest + n * r;
  return true;
}

}  // namespace

ContactMask step(World& world, std::span<const Paddle> paddles, real dt) {
  BallState& ball = world.ball;
  const BallParams& bp = world.ball_params;
  ContactMask mask = contact::kNone;

  BallState prev = ball;
  real remaining = dt;
  step_rk4(ball, bp, remaining);

  // After an impact at fraction f: restart the flight from the impact state
  // for whatever is left of the step.
  const auto resume = [&](real f) {
    prev = ball;
    remaining *= real{1} - f;
    step_rk4(ball, bp, remaining);
  };

  for (std::size_t i = 0; i < paddles.size() && i < 2; ++i) {
    if (!paddles[i].enabled) continue;
    Paddle paddle = paddles[i];
    paddle.pos += paddle.vel * (dt - remaining);  // where it is at the sub-step start
    if (const auto f = collide_paddle(ball, bp, prev, paddle, remaining)) {
      mask |= i == 0 ? contact::kPaddle0 : contact::kPaddle1;
      resume(*f);
    }
  }

  if (collide_net(ball, bp, prev, world.table, world.net)) mask |= contact::kNet;

  const Table& t = world.table;
  const real r = bp.radius;
  if (const auto f = collide_horizontal(ball, bp, prev, t.height, t.width * 0.5f,
                                        t.length * 0.5f, t.surface)) {
    mask |= ball.pos.z >= 0 ? contact::kTableNear : contact::kTableFar;
    resume(*f);
    ball.pos.y = std::max(ball.pos.y, t.height + r);  // a resting ball must not sink
  } else if (const auto g = collide_horizontal(ball, bp, prev, real{0}, kInfinity, kInfinity,
                                               world.floor.surface)) {
    mask |= contact::kFloor;
    resume(*g);
    ball.pos.y = std::max(ball.pos.y, r);
  }

  return mask;
}

}  // namespace pp
