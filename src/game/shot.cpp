#include "game/shot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace pp {

namespace {

constexpr real kPlanDt = 2e-3f;  // planning can afford a coarser step than the game

real horizontal_distance(Vec3 a, Vec3 b) {
  return std::hypot(b.x - a.x, b.z - a.z);
}

// The paddles play from behind the end lines, so a ball that would bounce
// twice on the table can never be returned. Every planned ball must therefore
// "carry": after its bounce on the receiving half it has to pass the end line
// at a playable height before it comes down again.
constexpr real kCarryPast = 0.15f;  // metres beyond the end line

enum class Outcome { Pending, Good, Bad };

// Shared tail of both checks: called every step once the ball has bounced on
// the receiving half.
Outcome carry_status(const World& world, ContactMask c) {
  if (c & (contact::kTableNear | contact::kTableFar | contact::kNet)) return Outcome::Bad;
  const Vec3 p = world.ball.pos;
  if (std::abs(p.z) > world.table.length * 0.5f + kCarryPast) {
    return p.y > world.table.height ? Outcome::Good : Outcome::Bad;
  }
  return (c & contact::kFloor) ? Outcome::Bad : Outcome::Pending;
}

// Plays a serve forward with the real collision code and checks the sequence
// of contacts: own half, then (without touching the net) the other half, then
// out past the end line.
bool serve_is_legal(World world, const BallState& start) {
  world.ball = start;
  const ContactMask own = start.pos.z > 0 ? contact::kTableNear : contact::kTableFar;
  const ContactMask other = start.pos.z > 0 ? contact::kTableFar : contact::kTableNear;
  const real half_w = world.table.width * 0.5f;
  const real half_l = world.table.length * 0.5f;

  bool bounced_own = false, bounced_other = false;
  for (int i = 0; i < 2500; ++i) {
    const ContactMask c = step(world, {}, 1e-3f);
    if (bounced_other) {
      const Outcome o = carry_status(world, c);
      if (o != Outcome::Pending) return o == Outcome::Good;
      continue;
    }
    if (c & (contact::kNet | contact::kFloor)) return false;
    if (c & own) {
      if (bounced_own) return false;  // twice on the server's side
      bounced_own = true;
    }
    if (c & other) {
      // Keep away from the edges so small differences in play do not turn a
      // planned serve into a fault.
      const Vec3 p = world.ball.pos;
      if (!bounced_own || std::abs(p.x) > half_w - 0.06f || std::abs(p.z) > half_l - 0.10f) {
        return false;
      }
      bounced_other = true;
    }
  }
  return false;
}

}  // namespace

// Same idea for a rally stroke: straight to the other half, then out the back.
bool stroke_carries(World world, const BallState& start) {
  world.ball = start;
  const ContactMask other = start.pos.z > 0 ? contact::kTableFar : contact::kTableNear;
  bool bounced = false;
  for (int i = 0; i < 2500; ++i) {
    const ContactMask c = step(world, {}, 1e-3f);
    if (bounced) {
      const Outcome o = carry_status(world, c);
      if (o != Outcome::Pending) return o == Outcome::Good;
    } else if (c == other) {
      bounced = true;
    } else if (c != contact::kNone) {
      return false;
    }
  }
  return false;
}

Flight fly(const World& world, BallState s, real max_time) {
  const real r = world.ball_params.radius;
  const real land_y = world.table.height + r;
  const real net_top = world.table.height + world.net.height;

  Flight out;
  for (real t = 0; t < max_time; t += kPlanDt) {
    const BallState prev = s;
    step_rk4(s, world.ball_params, kPlanDt);

    if ((prev.pos.z > 0) != (s.pos.z > 0)) {
      const real u = prev.pos.z / (prev.pos.z - s.pos.z);
      out.crossed_net = true;
      out.net_clearance = lerp(prev.pos, s.pos, u).y - r - net_top;
    }
    if (s.vel.y < 0 && prev.pos.y > land_y && s.pos.y <= land_y) {
      const real u = (prev.pos.y - land_y) / (prev.pos.y - s.pos.y);
      out.landing = lerp(prev.pos, s.pos, u);
      out.time = t + u * kPlanDt;
      out.landed = true;
      return out;
    }
  }
  return out;
}

std::optional<Vec3> aim(const World& world, Vec3 from, Vec3 spin, Vec3 target, real flight_time) {
  // First guess: the vacuum solution of  x(T) = from + v T + g T^2 / 2.
  const Vec3 g = world.ball_params.gravity;
  Vec3 v = (target - from) / flight_time - g * (flight_time * 0.5f);

  // Then fly it, see where it really lands, and push the horizontal velocity
  // by the miss. Converges in three or four rounds.
  for (int i = 0; i < 8; ++i) {
    const Flight f = fly(world, {.pos = from, .vel = v, .spin = spin});
    if (!f.landed || f.time <= 0) return std::nullopt;
    const real miss_x = target.x - f.landing.x;
    const real miss_z = target.z - f.landing.z;
    if (miss_x * miss_x + miss_z * miss_z < 1e-4f) break;  // within 1 cm
    v.x += miss_x / f.time;
    v.z += miss_z / f.time;
  }
  return v;
}

std::optional<Vec3> plan_shot(const World& world, Vec3 from, Vec3 spin, Vec3 target, real speed) {
  const real deepest = world.table.length * 0.5f - 0.10f;

  std::optional<Vec3> best;
  for (int attempt = 0; attempt < 5; ++attempt) {
    real flight_time = std::max(0.12f, horizontal_distance(from, target) / std::max(speed, 1.0f));

    // Start as flat as the requested speed allows; add loft until it clears.
    for (int i = 0; i < 14; ++i) {
      const auto v = aim(world, from, spin, target, flight_time);
      if (v) {
        const Flight f = fly(world, {.pos = from, .vel = *v, .spin = spin});
        best = v;
        if (f.landed && f.crossed_net && f.net_clearance >= 0.035f) break;
      }
      flight_time *= 1.12f;
    }

    // A soft, short ball would die on the table. Aim deeper and try again.
    if (best && stroke_carries(world, {.pos = from, .vel = *best, .spin = spin})) return best;
    if (std::abs(target.z) >= deepest) break;
    target.z = std::copysign(std::min(std::abs(target.z) + 0.14f, deepest), target.z);
  }
  return best;
}

std::optional<ServePlan> plan_serve(const World& world, Vec3 from, real aim_x, real pace) {
  const real r = world.ball_params.radius;
  const real half_l = world.table.length * 0.5f;
  const real own = from.z > 0 ? real{1} : real{-1};
  const real wanted_time = 0.40f + (0.10f - 0.40f) * std::clamp(pace, real{0}, real{1});

  // Candidates: where on the server's half the first bounce goes, how long
  // the ball takes to get there, and how much topspin it carries. Each one is
  // tested by simulation.
  //
  // Two regimes have to be covered. Struck high, the ball can be driven down
  // hard and short (small time, bounce near the server). Struck low, it must
  // be lifted so that it falls from enough height to bounce over the net,
  // which takes longer and puts the first bounce close to the net.
  constexpr std::array<real, 6> kFirstBounce{0.70f, 0.55f, 0.40f, 0.28f, 0.18f, 0.85f};  // of the half
  constexpr std::array<real, 13> kTimes{0.09f, 0.11f, 0.13f, 0.15f, 0.18f, 0.21f, 0.25f,
                                        0.29f, 0.34f, 0.40f, 0.46f, 0.53f, 0.60f};
  constexpr std::array<real, 4> kTopspin{0.0f, 60.0f, 120.0f, 200.0f};  // rad/s

  std::optional<ServePlan> best;
  real best_score = std::numeric_limits<real>::infinity();
  for (const real t : kTimes) {
    const real score = std::abs(t - wanted_time);
    if (score >= best_score) continue;  // cannot beat what we already have
    bool found = false;
    for (std::size_t si = 0; si < kTopspin.size() && !found; ++si) {
      // Travelling toward -z (server at +z), topspin is about -x.
      const Vec3 spin{-own * kTopspin[si], 0, 0};
      for (const real frac : kFirstBounce) {
        const Vec3 first{from.x + (aim_x - from.x) * 0.35f, world.table.height + r,
                         own * frac * half_l};
        if ((from.z - first.z) * own < 0.05f) continue;  // must travel toward the net
        const auto v = aim(world, from, spin, first, t);
        if (!v) continue;
        if (!serve_is_legal(world, {.pos = from, .vel = *v, .spin = spin})) continue;
        best = ServePlan{*v, spin};
        best_score = score;
        found = true;
        break;
      }
    }
  }
  return best;
}

std::optional<Crossing> predict_crossing(const World& world, Vec3 plane_point, Vec3 toward_ball,
                                         bool already_bounced, real horizon) {
  World w = world;  // simulate a private copy
  // The ball comes from -z when the plane faces -z: that is the player's end.
  const ContactMask receiving = toward_ball.z < 0 ? contact::kTableNear : contact::kTableFar;
  const auto side = [&](Vec3 p) { return dot(p - plane_point, toward_ball); };

  Crossing out;
  out.bounced = already_bounced;
  for (real t = 0; t < horizon; t += kPlanDt) {
    const Vec3 prev = w.ball.pos;
    const ContactMask c = step(w, {}, kPlanDt);
    if (c & contact::kFloor) return std::nullopt;
    if (c & receiving) {
      if (out.bounced) return std::nullopt;  // second bounce: dead before it arrives
      out.bounced = true;
    }

    const real d0 = side(prev), d1 = side(w.ball.pos);
    if (d0 > 0 && d1 <= 0) {
      const real u = d0 / (d0 - d1);
      out.point = lerp(prev, w.ball.pos, u);
      out.time = t + u * kPlanDt;
      return out;
    }
  }
  return std::nullopt;
}

}  // namespace pp
