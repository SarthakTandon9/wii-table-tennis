#pragma once
// From camera observations to the player's bat: the whole of hand control in
// one place, so that the game and the replay tool run exactly the same code.
//
// The division of labour is the one motion-controller games use:
//
//   the BODY is the game's   It walks to where the ball will arrive, at a
//                            limited speed, and comes back to the middle
//                            after each shot. The view moves with it, so it
//                            reads as footwork.
//   the ARM is the player's  The hand moves the bat within arm's reach of
//                            the body, directly: no freeze, next to no
//                            smoothing, and prediction to cancel the camera's
//                            delay.
//
// Why. Recorded play with the earlier design (hand mapped across the whole
// table, a long-range magnet to finish the job) showed the magnet moving the
// bat a median 40 cm per ball, the bat frozen during every swing, and about
// 200 ms between the hand moving and the bat following. Research on control
// puts the limit for "this is me doing it" at roughly 100 to 125 ms, and
// finds that help toward a target is only invisible while it is modest. The
// player's bat was visibly being flown by the game.
//
// Shrinking the hand's range from the table (1.5 m) to an arm (0.64 m) is
// also what makes directness affordable: the same camera noise is spread
// over less than half the distance, so the heavy filtering that hid it, and
// the lag that filtering cost, can go.
//
// The swing. While the hand swings, two things are true at once:
//  - for the RULES the bat stays where it was just before the swing (see
//    bat_anchor.hpp): whether the ball is reached must not depend on which
//    way the swing happened to go;
//  - on SCREEN the bat goes where the hand goes. `display_offset` is that
//    movement, added only when drawing.

#include "core/types.hpp"
#include "estimation/palm_filter.hpp"
#include "game/bat_anchor.hpp"
#include "game/bat_face.hpp"
#include "game/bat_smoother.hpp"
#include "game/swing.hpp"
#include "math/vec3.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

struct HandControlConfig {
  // The body.
  // 2.1 m/s. At 2.4 the body reached every one of 51 balls in a recorded
  // session, so nothing the opponent did with placement could matter.
  real body_speed = 2.1f;     // m/s (simulation time): a quick side-step
  real body_accel = 14.0f;    // m/s^2
  real body_limit = 0.95f;    // m either side of the table's centre line
  real lean_boost = 0.35f;    // extra speed when the hand already leans the right way
  real ready_lean = 0.5f;     // between balls, the body drifts this share of the way the hand leans
  // Going for a ball, the body allows for where the arm is: with the hand
  // held out to one side it stands that much less far over, by this share.
  // At 1 the bat would land on the ball wherever the hand was (the hand
  // would mean nothing); at 0 a hand resting off-centre misses every ball by
  // that much. Replaying recorded play, 0 left the bat off to the side on 7
  // to 11 balls a session.
  real arm_allowance = 0.5f;
  // What is shown of a swing on top of the hand's own movement: the bat also
  // goes forward, toward the table, by up to this much.
  real lunge = 0.16f;         // m
  real lunge_full_speed = 3.0f;  // image widths per second for the full lunge
};

inline constexpr real kArmReach = 0.32f;  // m: the hand's range either side of the body

// The hand's mapping for arm-relative control.
inline PalmFilterConfig arm_filter_config() {
  PalmFilterConfig c;
  c.world_half_width = kArmReach;
  // With the hand's range this small the prediction is worth having again:
  // 70 ms covers the camera's delay as measured in play (64 ms).
  c.max_lookahead = 0.07;
  return c;
}

// Light: at this range the camera's noise is a couple of millimetres.
inline constexpr BatSmootherConfig kArmSmoothing{.min_cutoff = 3.0f, .beta = 3.0f,
                                                 .speed_cutoff = 3.0f};

// What the game knows about the incoming ball, if there is one to go for.
struct BallCue {
  bool incoming = false;  // a ball the player may hit is on its way
  real meeting_x = 0;     // where it will cross the bat's plane
  real due_in = 0;        // simulation seconds until it does (negative: just gone by)
};

struct HandControlOutput {
  bool hand_seen = false;   // false: no usable hand right now (the bat keeps its place)
  Vec3 bat;                 // where the bat is for the rules (body + arm, anchored in a swing)
  Vec3 vel;                 // the hand's velocity in the world, m/s of real time
  Vec3 display_offset;      // add to `bat` when drawing: the swing as the hand makes it
  real body_x = 0;
  Vec3 swing_velocity;      // image widths per second; what the swing detector saw
  bool swing_began = false; // on this update
  bool holding = false;     // the rules' bat is anchored (a swing is under way)
  real lag_free_arm_x = 0;  // the arm's x before smoothing (for measuring lag)
};

class HandControl {
 public:
  explicit HandControl(HandControlConfig config = {},
                       const PalmFilterConfig& filter = arm_filter_config(),
                       BatSmootherConfig smoothing = kArmSmoothing)
      : cfg_(config), filter_(filter), smoother_(smoothing) {}

  void observe(const HandObservation& obs) { filter_.observe(obs); }

  // `now` on the host clock; `elapsed` real seconds since the last call;
  // `sim_elapsed` simulation seconds stepped since the last call.
  HandControlOutput update(double now, real elapsed, real sim_elapsed, const BallCue& cue) {
    HandControlOutput out;
    const HandPose hand = filter_.pose_at(now);
    out.hand_seen = hand.valid;

    // ---- swinging? ----
    // Judged by the hand's own speed in the image, in any direction. With a
    // ball about to arrive, the quicker of two speed estimates is used and
    // the bar is lower: see kEagerSwing.
    const bool ball_due = cue.incoming && cue.due_in > -0.25f && cue.due_in < 0.35f;
    if (hand.valid) {
      const Vec3 steady{hand.image_vel.x, hand.image_vel.y, kPushWeight * hand.image_vel.z};
      const Vec3 quick{hand.image_vel_fast.x, hand.image_vel_fast.y,
                       kPushWeight * hand.image_vel_fast.z};
      out.swing_velocity = ball_due && length_sq(quick) > length_sq(steady) ? quick : steady;
    }
    const real swing_speed = length(out.swing_velocity);
    out.swing_began = swing_.update(out.swing_velocity, elapsed, ball_due ? kEagerSwing : real{1});

    // ---- the body ----
    // Toward the ball when one is coming; otherwise back toward the middle,
    // shaded the way the hand leans (lean left and the player stands left).
    const real lean = std::clamp(arm_.x / kArmReach, -1.0f, 1.0f);
    const real target = cue.incoming ? cue.meeting_x - cfg_.arm_allowance * arm_.x
                                     : cfg_.ready_lean * lean * cfg_.body_limit;
    const real gap = std::clamp(target, -cfg_.body_limit, cfg_.body_limit) - body_x_;
    // A player whose hand is already heading that way gets there sooner.
    const bool leaning_right_way = cue.incoming && lean * gap > 0.05f;
    const real top = cfg_.body_speed * (1.0f + (leaning_right_way ? cfg_.lean_boost : 0.0f));
    // Brake in time: the speed from which the remaining gap can still be stopped in.
    const real stoppable = std::sqrt(2.0f * cfg_.body_accel * std::abs(gap));
    const real wanted = std::copysign(std::min(top, stoppable), gap);
    const real dv = cfg_.body_accel * sim_elapsed;
    body_v_ += std::clamp(wanted - body_v_, -dv, dv);
    body_x_ += body_v_ * sim_elapsed;
    out.body_x = body_x_;

    // ---- the arm ----
    if (hand.valid) {
      out.lag_free_arm_x = hand.pos.x;
      // For the rules: held still through a swing, lightly smoothed otherwise.
      arm_ = smoother_.update(anchor_.update(hand.pos, swing_speed, swing_.live(), elapsed), elapsed);
      // For the eye: during a swing, how far the hand itself has gone from
      // there. (Outside a swing the two are the same bat, and drawing the
      // unsmoothed hand would only put the shake back.)
      const bool swinging = anchor_.holding() || swing_.live();
      const Vec3 moved = swinging ? hand.pos - arm_ : Vec3{};
      shown_ += (moved - shown_) * (1.0f - std::exp(-elapsed / (swinging ? 0.020f : 0.090f)));
      const real reach = swinging ? std::clamp(swing_speed / cfg_.lunge_full_speed, 0.0f, 1.0f) : 0.0f;
      lunge_ += (reach - lunge_) * (1.0f - std::exp(-elapsed / (reach > lunge_ ? 0.03f : 0.12f)));
      face_.update(hand.roll, hand.tilt, swinging, elapsed);
      hand_vel_ = hand.vel;
    } else {
      // Hand not visible this moment: the arm stays as it was.
      shown_ *= std::exp(-elapsed / 0.10f);
      lunge_ *= std::exp(-elapsed / 0.12f);
      hand_vel_ = {};
    }
    out.bat = {body_x_ + arm_.x, arm_.y, arm_.z};
    out.vel = hand_vel_;
    out.display_offset = shown_ + Vec3{0, 0, -cfg_.lunge * lunge_};
    out.holding = anchor_.holding();
    return out;
  }

  void reset() { *this = HandControl{cfg_, filter_.config(), smoother_.config()}; }

  SwingDetector& swing() { return swing_; }
  const SwingDetector& swing() const { return swing_; }
  const BatFace& face() const { return face_; }
  const PalmFilter& filter() const { return filter_; }

 private:
  HandControlConfig cfg_;
  PalmFilter filter_;
  SwingDetector swing_{kHandSwing};
  BatAnchor anchor_;
  BatSmoother smoother_;
  BatFace face_;
  Vec3 arm_{0, 0.95f, 1.50f};  // bat relative to the body (x), absolute height and depth
  Vec3 shown_, hand_vel_;
  real lunge_ = 0;
  real body_x_ = 0, body_v_ = 0;
};

}  // namespace pp
