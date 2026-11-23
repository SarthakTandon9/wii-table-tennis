#pragma once
// Where and when the incoming ball will reach the player's bat.
//
// The answer comes from a forward simulation of the ball (predict_crossing),
// which is by far the most expensive thing the simulation thread does: one
// call steps the physics a few hundred times.
//
// Two ways to keep the answer current:
//
//   Periodic   run the forward simulation every few wake-ups. Simple, and
//              what the game did at first (every 4th wake-up, 250 a second).
//
//   OnEvent    run it only when the answer can have changed. Between
//              bounces the ball is in free flight, and free flight is
//              deterministic: the meeting point computed at the start of the
//              flight is still right at the end of it. Only the time
//              remaining changes, and that is a subtraction. So predict once
//              when the ball is struck, again whenever something deflects it
//              (table, net), and otherwise just count down.
//
// "Something deflected it" is detected from the ball itself: in flight its
// velocity changes smoothly (gravity and drag, well under 0.1 m/s per
// millisecond), so a larger change in one step means it hit something.
// A slow periodic refresh stays in as a safety net.

#include "core/types.hpp"
#include "game/match.hpp"
#include "game/shot.hpp"
#include "math/vec3.hpp"

#include <algorithm>

namespace pp {

struct MeetingPoint {
  bool known = false;     // the ball is coming and will reach the bat's plane
  bool hittable = false;  // ...after bouncing on the player's half (legal to hit)
  Vec3 point;
  real time = 0;          // simulation seconds from now; briefly negative once it has arrived
};

class MeetingPointTracker {
 public:
  enum class Policy { Periodic, OnEvent };

  explicit MeetingPointTracker(Policy policy = Policy::OnEvent) : policy_(policy) {}

  // Call once per wake-up, after `elapsed` simulation seconds have been
  // stepped since the last call. The plane is given by a point on it and its
  // unit normal toward the incoming ball.
  MeetingPoint update(const Match& match, Vec3 plane_point, Vec3 toward_ball, real elapsed) {
    const bool incoming = match.phase == Phase::Rally && match.striker == Side::Opponent &&
                          match.stage != Stage::ServeToss;
    if (!incoming) {
      current_ = {};
      was_incoming_ = false;
      return current_;
    }

    const Vec3 vel = match.world.ball.vel;
    ++wakes_since_;
    bool refresh;
    if (policy_ == Policy::Periodic) {
      refresh = !was_incoming_ || wakes_since_ >= kPeriod;
    } else {
      const bool deflected = length_sq(vel - last_vel_) > kDeflection * kDeflection;
      const bool plane_moved = length_sq(plane_point - plane_point_) > 1e-8f ||
                               length_sq(toward_ball - toward_ball_) > 1e-8f;
      // Once the ball has gone by the plane there is nothing left to predict:
      // a forward simulation from there finds no crossing, and acting on that
      // would drop the ball in the middle of the window in which a late
      // swing may still strike it. (The periodic safety refresh did exactly
      // that in recorded play: swings 10 ms late missed a ball 5 cm away.)
      const bool gone_by = current_.known && current_.time < 0;
      refresh = !was_incoming_ || deflected || match.stage != last_stage_ ||
                (!gone_by && (plane_moved || wakes_since_ >= kSafetyRefresh));
    }
    last_vel_ = vel;
    last_stage_ = match.stage;
    was_incoming_ = true;

    if (refresh) {
      ++predictions_;
      wakes_since_ = 0;
      plane_point_ = plane_point;
      toward_ball_ = toward_ball;
      const auto c = predict_crossing(match.world, plane_point, toward_ball,
                                      match.stage == Stage::AwaitHit);
      current_ = c ? MeetingPoint{true, c->bounced, c->point, c->time} : MeetingPoint{};
      plane_depth_ = current_.point.z;
    } else if (current_.known && match.strike_hold <= 0) {
      current_.time -= elapsed;  // free flight: only the clock has moved
    }                            // (and not even that while the ball is held on the bat)
    if (current_.known && current_.time < 0) {
      // The ball has reached the bat's plane and is going by. For a short
      // while longer it can still be struck (a swing registers late), so keep
      // reporting it, now AT the ball: the bat should stay with the ball
      // instead of springing back to the hand. In recorded play the magnet
      // let go at this instant, and swings that began 10 ms later missed
      // because the bat had already slid 25 cm away.
      if (current_.time < -kLateWindow) {
        current_ = {};
      } else {
        // The bat reaches back with it, up to a point (it should not end
        // up in the player's lap), so a late hit looks like one.
        current_.point.x = match.world.ball.pos.x;
        current_.point.y = match.world.ball.pos.y;
        current_.point.z = std::min(match.world.ball.pos.z, plane_depth_ + kLateReachBack);
      }
    }
    return current_;
  }

  long predictions() const { return predictions_; }

 private:
  static constexpr int kPeriod = 4;            // Periodic: wake-ups between predictions
  static constexpr int kSafetyRefresh = 150;   // OnEvent: re-check at least this often
  static constexpr real kDeflection = 0.5f;    // m/s in one wake-up: it hit something
  static constexpr real kLateWindow = 0.22f;   // s after arrival that the ball is still tracked
  static constexpr real kLateReachBack = 0.40f;  // m the bat follows the ball back past its plane

  Policy policy_;
  MeetingPoint current_;
  Vec3 last_vel_, plane_point_, toward_ball_;
  Stage last_stage_ = Stage::ServeToss;
  bool was_incoming_ = false;
  real plane_depth_ = 0;  // z of the predicted meeting point, before any reaching back
  int wakes_since_ = 0;
  long predictions_ = 0;
};

}  // namespace pp
