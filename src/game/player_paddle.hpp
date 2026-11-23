#pragma once
// The player's paddle: where the input says it is, plus the help that makes
// the ball hittable.
//
// The honest problem: the player is aiming a 20 cm bat at a 4 cm ball through
// a flat screen, with a hand the camera sees late and imprecisely. Nobody can
// do that unaided, and games with motion controls do not ask them to. The
// standard remedy is a "magnet": when the ball is about to arrive and the
// paddle is roughly in the right place, the paddle is drawn toward the exact
// meeting point. Roughly right becomes a hit; clearly wrong is still a miss.
//
// The pull is designed to be felt as "I got it" and not as the game playing
// itself:
//  - it only starts shortly before the ball arrives, and grows as it nears;
//  - it fades with distance, so a paddle on the wrong side gets none;
//  - it moves the paddle's position only. The velocity given to the physics
//    is the player's own, so how hard and where the ball goes is still theirs.

#include "core/types.hpp"
#include "math/vec3.hpp"

namespace pp {

struct MagnetConfig {
  bool enabled = true;
  // Reach is sized from recorded play: the unaided paddle was within 29 cm
  // of the ball 80% of the time and within 58 cm 95% of the time. Getting
  // the bat there is not meant to be the hard part; the swing is.
  real max_pull = 0.92f;        // share of the gap closed at full strength
  real reach_full = 0.40f;      // m: within this, full strength
  real reach_none = 0.80f;      // m: beyond this, none
  real time_start = 0.60f;      // s before arrival: pull begins
  real time_full = 0.12f;       // s before arrival: pull at full strength
  real ease = 0.035f;           // s: how quickly the paddle slides toward the pull
};

// Mouse: the cursor is precise, so the magnet only finishes the job.
inline constexpr MagnetConfig kMouseMagnet{};
// Hand: a nudge, no more. Getting to the ball is the body's job (see
// game/hand_control.hpp); the magnet only hides what is left of the tracking
// error at the last moment, within 20 cm, fading out by 45. It was once the
// whole of positioning, with a reach of 1.45 m, and recorded play showed it
// moving the bat a median 40 cm per ball: the bat visibly flew itself.
inline constexpr MagnetConfig kHandMagnet{.enabled = true, .max_pull = 0.90f, .reach_full = 0.20f,
                                          .reach_none = 0.45f, .time_start = 0.35f,
                                          .time_full = 0.10f, .ease = 0.035f};

// How strongly to pull (0..1), given how far the paddle is from the meeting
// point and how soon the ball gets there. Exposed for tests.
real magnet_strength(const MagnetConfig& cfg, real distance, real time_to_arrival);

class PlayerPaddle {
 public:
  MagnetConfig magnet;

  // Input arriving in steps (mouse events): head for `target`, taking about
  // `step_seconds` of simulation time to get there.
  void step_toward(Vec3 target, real step_seconds);

  // Input arriving continuously (the filtered hand): be exactly here, moving
  // at `velocity` (simulation time).
  void place(Vec3 position, Vec3 velocity);

  // Where and when the ball will meet the paddle's plane. `pull_depth` also
  // draws the paddle along z (used for the serve toss).
  void set_meeting_point(Vec3 point, real time_to_arrival, bool pull_depth = false);
  void clear_meeting_point() { has_meeting_ = false; }

  // One simulation tick. `wall_dt` is the real time that tick represents.
  void advance(real dt, real wall_dt);

  Vec3 pos() const { return raw_ + offset_; }  // where the paddle is, pull included
  Vec3 raw_pos() const { return raw_; }        // where the input alone would put it
  Vec3 vel() const { return smoothed_; }       // what the physics is told
  real pull() const { return pull_; }          // current magnet strength, 0..1
  // Simulation seconds until the ball reaches the meeting point; negative
  // once it has passed. Only meaningful while has_meeting_point().
  bool has_meeting_point() const { return has_meeting_; }
  real time_to_arrival() const { return time_to_arrival_; }

 private:
  Vec3 raw_{0, 1.0f, 1.50f};
  Vec3 target_ = raw_;
  Vec3 travel_;      // m/s (simulation time)
  Vec3 smoothed_;
  Vec3 offset_;      // the magnet's displacement
  bool stepping_ = false;

  bool has_meeting_ = false;
  bool pull_depth_ = false;
  Vec3 meeting_;
  real time_to_arrival_ = 0;
  real pull_ = 0;
};

}  // namespace pp
