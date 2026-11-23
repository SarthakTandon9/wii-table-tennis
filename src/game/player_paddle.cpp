#include "game/player_paddle.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

namespace {

constexpr real kVelSmoothing = 0.030f;   // s (real time): averages out input jitter
constexpr real kMaxPaddleSpeed = 9.0f;   // m/s handed to the physics
constexpr real kMaxTravelSpeed = 25.0f;  // m/s the paddle may actually move

// 0 at `from`, 1 at `to`, smooth in between (and clamped outside).
real smoothstep(real from, real to, real x) {
  const real t = std::clamp((x - from) / (to - from), real{0}, real{1});
  return t * t * (3.0f - 2.0f * t);
}

Vec3 capped(Vec3 v, real limit) {
  const real s = length(v);
  return s > limit ? v * (limit / s) : v;
}

}  // namespace

real magnet_strength(const MagnetConfig& cfg, real distance, real time_to_arrival) {
  if (!cfg.enabled) return 0;
  const real by_distance = 1.0f - smoothstep(cfg.reach_full, cfg.reach_none, distance);
  const real by_time = 1.0f - smoothstep(cfg.time_full, cfg.time_start, time_to_arrival);
  return cfg.max_pull * by_distance * by_time;
}

void PlayerPaddle::step_toward(Vec3 target, real step_seconds) {
  target_ = target;
  travel_ = capped((target_ - raw_) / std::max(step_seconds, real{1e-3f}), kMaxTravelSpeed);
  stepping_ = true;
}

void PlayerPaddle::place(Vec3 position, Vec3 velocity) {
  raw_ = position;
  target_ = position;
  travel_ = velocity;
  stepping_ = false;
}

void PlayerPaddle::set_meeting_point(Vec3 point, real time_to_arrival, bool pull_depth) {
  meeting_ = point;
  time_to_arrival_ = time_to_arrival;
  pull_depth_ = pull_depth;
  has_meeting_ = true;
}

void PlayerPaddle::advance(real dt, real wall_dt) {
  if (stepping_) {
    const Vec3 remaining = target_ - raw_;
    const Vec3 move = travel_ * dt;
    // Stop at the target instead of sailing past it.
    if (dot(remaining, move) <= 0 || length_sq(move) >= length_sq(remaining)) {
      raw_ = target_;
      travel_ = {};
    } else {
      raw_ += move;
    }
  }
  smoothed_ += (capped(travel_, kMaxPaddleSpeed) - smoothed_) *
               (1.0f - std::exp(-wall_dt / kVelSmoothing));

  // The magnet. Its target displacement is recomputed every tick from the
  // RAW position, so a hand that moves away takes the paddle with it.
  Vec3 wanted{};
  pull_ = 0;
  if (has_meeting_) {
    time_to_arrival_ -= dt;
    if (time_to_arrival_ < -0.30f) {
      has_meeting_ = false;  // the ball has gone by
    } else {
      Vec3 gap = meeting_ - raw_;
      if (!pull_depth_) gap.z = 0;
      pull_ = magnet_strength(magnet, length(gap), std::max(time_to_arrival_, real{0}));
      wanted = gap * pull_;
    }
  }
  // Slide there, do not snap: a paddle that teleports looks like a glitch.
  offset_ += (wanted - offset_) * (1.0f - std::exp(-dt / magnet.ease));
}

}  // namespace pp
