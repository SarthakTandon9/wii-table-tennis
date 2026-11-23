#pragma once
// Recognising a swing.
//
// Getting the bat to the ball is something the game helps with. The swing is
// the player's own act: when it happens, how hard, and in which direction
// decide whether the ball is hit and where it goes. This is the model motion
// games use. It does not ask for a realistic stroke; any sharp movement of
// the hand counts, as long as it comes at the right moment.
//
// A swing is a burst of speed. It starts when the hand's speed rises above
// `start_speed`, and the bat is then "live" (able to strike the ball) for
// `live_for` seconds. Another swing cannot start until the hand has slowed
// below `rearm_speed`, so holding the hand in constant motion does not count
// as swinging continuously, and after a swing that hit nothing there is a
// short recovery, as there is with a real bat.
//
// Speeds are in whatever unit the caller's velocities use: image widths per
// second for the tracked hand, metres per second for the mouse.

#include "core/types.hpp"
#include "math/vec3.hpp"

#include <algorithm>

namespace pp {

struct SwingConfig {
  // Defaults are for the tracked hand, in image widths per second. In a
  // recorded session with no swinging at all, moving the bat into position
  // reached 0.67 (90th percentile) and 1.6 at the very fastest. The start
  // speed sits near the top of that, so positioning rarely reads as a swing;
  // at arm's length from a laptop camera it is a hand moving about 1.2 m/s.
  // Recordings that did include swings then showed real ones at 2.0 to 4.0
  // sideways, and forward pushes of similar size.
  real start_speed = 1.3f;    // a swing begins above this
  real rearm_speed = 0.7f;    // and another may begin only after dropping below this
  // Full power sits near the top of what was recorded, not the middle, and
  // not above it either. Recorded peaks over three sessions: 1.3 (10th
  // percentile), 2.3 (median), 3.5 to 3.6 (75th to 90th). At 2.8, four hits
  // in ten scored as flat-out and hitting harder changed nothing; at 4.5 the
  // median hit scored 0.19 and every stroke was a tap.
  real full_speed = 3.6f;     // peak speed that counts as a full-power swing
  // Recorded swings came anywhere from 0.95 s early to 0.23 s late. The bat
  // stays live long enough for a swing begun well ahead of the ball.
  // 0.40 s: in the latest recordings swings began 0.11 to 0.18 s before the
  // ball was due at the median and 0.42 s at the 90th percentile. (It was
  // 0.55 s while hitting at all was the difficulty. With every ball being
  // returned, 51 of 51 in one session, a swing thrown half a second early
  // should not count as timing the ball.)
  real live_for = 0.40f;      // s: how long the bat can strike after the swing starts
  real recovery = 0.15f;      // s after an unused swing before the next can start
};

// Suggested starting points; both get tuned from recorded sessions.
inline constexpr SwingConfig kHandSwing{};
inline constexpr SwingConfig kMouseSwing{.start_speed = 1.6f, .rearm_speed = 0.7f,
                                         .full_speed = 5.0f, .live_for = 0.55f, .recovery = 0.15f};

// How much a push toward the camera counts, next to sideways movement, when
// judging the hand's speed. Recorded pushes ran at 1.7 to 3.1 (growth rate of
// the hand's apparent size, per second); this puts a firm one over the line.
inline constexpr real kPushWeight = 0.7f;

// While a ball is due, a swing begins at this share of the usual speed.
inline constexpr real kEagerSwing = 0.77f;  // 1.0 image widths per second for the hand

class SwingDetector {
 public:
  explicit SwingDetector(SwingConfig config = {}) : cfg_(config) {}

  // Call regularly with the current velocity and the real time since the last
  // call. Returns true on the update in which a swing begins.
  //
  // `eagerness` below 1 lowers the speed at which a swing begins. The caller
  // sets it while a ball is about to arrive: a fast movement at that moment
  // is almost certainly a swing, and recognising it a frame sooner matters
  // more than the odd mistake (which only means the ball gets tapped back).
  bool update(Vec3 velocity, real dt, real eagerness = 1) {
    const real speed = length(velocity);
    bool started = false;

    if (live_) {
      age_ += dt;
      if (speed > peak_speed_) {  // still accelerating: the swing's character is its peak
        peak_speed_ = speed;
        peak_velocity_ = velocity;
      }
      if (age_ >= cfg_.live_for) {
        live_ = false;
        lockout_ = cfg_.recovery;
      }
    } else {
      lockout_ = std::max(real{0}, lockout_ - dt);
      if (speed < cfg_.rearm_speed) armed_ = true;
      if (armed_ && lockout_ <= 0 && speed >= cfg_.start_speed * eagerness) {
        live_ = true;
        armed_ = false;
        age_ = 0;
        peak_speed_ = speed;
        peak_velocity_ = velocity;
        started = true;
      }
    }
    return started;
  }

  // The swing connected: it is over, and the next one may follow promptly.
  void consume() {
    live_ = false;
    lockout_ = 0;
  }

  bool live() const { return live_; }
  real age() const { return age_; }                // s since the swing began
  real peak_speed() const { return peak_speed_; }
  // 0 at the slowest movement that counts as a swing, 1 at a full one.
  real power() const {
    return std::clamp((peak_speed_ - cfg_.start_speed) / (cfg_.full_speed - cfg_.start_speed),
                      real{0}, real{1});
  }
  // Which way the hand was going at its fastest, as a unit vector.
  Vec3 direction() const { return normalized(peak_velocity_); }
  const SwingConfig& config() const { return cfg_; }

 private:
  SwingConfig cfg_;
  bool live_ = false, armed_ = true;
  real age_ = 0, lockout_ = 0, peak_speed_ = 0;
  Vec3 peak_velocity_;
};

}  // namespace pp
