#pragma once
// Keeping the bat where it was while the hand swings.
//
// The hand does two jobs: it places the bat, and it swings. With one camera
// both are read from the same movement, and they fight. A swing is a fast
// movement of 20 to 40 cm; mapped to the bat like any other movement it
// carries the bat across the whole table. Replaying recorded play showed
// exactly that: the bat sat 5 to 20 cm from the ball until a fifth of a
// second before it arrived, and then the player's own swing threw the bat up
// to 1.3 m away, before the swing had even registered as one.
//
// Controllers with motion sensors do not have this problem because they
// never use the swing for position at all: a swing is an acceleration over a
// threshold, and its direction is read separately. This does the same for a
// camera. Slow movement places the bat. Fast movement is a swing, and for as
// long as it lasts the bat stays at the place it was in just BEFORE the
// movement began.
//
// "Just before" matters: the speed estimate lags the position (it needs a
// few camera frames), so by the time the hand is known to be moving fast the
// bat has already been dragged part of the way. A short history of positions
// lets the anchor reach back past that.

#include "core/types.hpp"
#include "math/vec3.hpp"

#include <array>
#include <cmath>

namespace pp {

struct BatAnchorConfig {
  // Hand speeds, in the swing detector's unit (image widths per second).
  // Recorded positioning ran at 0.1 (median) to 1.0 (95th percentile);
  // swings at 1.3 to 5.
  real hold_speed = 1.0f;     // faster than this: stop following the hand
  real release_speed = 0.6f;  // slower than this: follow it again
  real lookback = 0.15f;      // s: hold the position from this long before
  real max_idle_hold = 0.30f; // s held with no swing live before following anyway
  real ease = 0.10f;          // s: how quickly the bat rejoins the hand afterwards
};

class BatAnchor {
 public:
  explicit BatAnchor(BatAnchorConfig config = {}) : cfg_(config) {}

  // Call once per wake-up with where the hand puts the bat, how fast the hand
  // is moving, whether a swing is live, and the real time since the last
  // call. Returns where the bat should be.
  Vec3 update(Vec3 hand_pos, real hand_speed, bool swing_live, real dt) {
    if (hand_speed < cfg_.release_speed) armed_ = true;

    const bool swing_began = swing_live && !was_swing_live_;
    was_swing_live_ = swing_live;
    if (!holding_ && ((armed_ && hand_speed >= cfg_.hold_speed) || swing_began)) {
      holding_ = true;
      armed_ = false;
      idle_ = 0;
      anchor_ = count_ > 0 ? looked_back() : hand_pos;
    }
    if (holding_) {
      idle_ = swing_live ? real{0} : idle_ + dt;
      if (!swing_live && (hand_speed < cfg_.release_speed || idle_ > cfg_.max_idle_hold)) {
        holding_ = false;
        offset_ = anchor_ - hand_pos;  // start from the anchor, then slide to the hand
      }
    }
    if (!holding_) offset_ *= std::exp(-dt / cfg_.ease);
    const Vec3 out = holding_ ? anchor_ : hand_pos + offset_;

    // History of where the bat has been (not the hand: a second fast movement
    // straight after a swing must not anchor to a point in mid-swing).
    since_sample_ += dt;
    if (since_sample_ >= kSampleEvery) {
      since_sample_ = 0;
      history_[static_cast<std::size_t>(head_)] = out;
      head_ = (head_ + 1) % kSamples;
      if (count_ < kSamples) ++count_;
    }
    return out;
  }

  bool holding() const { return holding_; }
  void reset() { *this = BatAnchor{cfg_}; }

 private:
  static constexpr real kSampleEvery = 0.01f;  // s
  static constexpr int kSamples = 32;          // 0.32 s of history

  Vec3 looked_back() const {
    const int want = static_cast<int>(cfg_.lookback / kSampleEvery);
    const int back = want < count_ ? want : count_ - 1;  // 0 = the newest sample
    return history_[static_cast<std::size_t>((head_ - 1 - back + 2 * kSamples) % kSamples)];
  }

  BatAnchorConfig cfg_;
  std::array<Vec3, kSamples> history_{};
  int head_ = 0, count_ = 0;
  real since_sample_ = 0;
  bool holding_ = false, armed_ = true, was_swing_live_ = false;
  real idle_ = 0;
  Vec3 anchor_, offset_;
};

}  // namespace pp
