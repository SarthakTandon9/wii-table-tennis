#pragma once
// The angle of the bat's face, from the hand.
//
// A bat is the palm of the hand extended: where the palm faces, the ball
// goes. Until now the face was made up from the bat's position. This reads it
// from the hand itself, as far as one camera allows:
//
//   turn (aim left or right)   from the hand's roll: how far the fingers
//       lean left or right in the picture. That is measured directly and
//       cleanly (about 1 degree of noise), and in recorded play it ranged
//       over 50 degrees either way, so it is something players do without
//       being asked. Fingers leaning left: the face turns left, the ball
//       goes left.
//
//   closing (over the ball)    from how much the palm is tipped out of the
//       picture plane, which shows as foreshortening. Tipped: the face
//       closes, the ball leaves lower with more topspin.
//
// What it cannot do: tell a palm tipped forward from one tipped back (they
// look the same to one camera), or see the palm turning about a vertical
// axis with a sign. A depth camera or a model that estimates landmark depth
// would lift both limits.
//
// The angles are smoothed here. They arrive in steps, 30 times a second, and
// a bat that changes its attitude in steps looks like it is twitching; and
// during a swing the hand blurs and the reading goes wild, so while the bat
// is held for a swing the face holds too.

#include "core/types.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

struct BatFaceConfig {
  real turn_per_roll = 0.40f;  // radians of face turn per radian of hand roll
  real max_turn = 0.45f;       // radians (26 degrees)
  real max_closed = 0.60f;     // radians (34 degrees)
  real smoothing = 0.07f;      // s
  // Nobody holds a hand dead upright: in recorded play the resting lean was
  // 10 to 17 degrees to the right. Taken at face value that would send every
  // ball right. So the turn is measured from the hand's own usual lean,
  // which is learned, slowly.
  real neutral_seconds = 12.0f;
};

class BatFace {
 public:
  explicit BatFace(BatFaceConfig config = {}) : cfg_(config) {}

  // `roll` and `tilt` as in HandPose. `hold` freezes the face (during a swing).
  void update(real roll, real tilt, bool hold, real dt) {
    if (hold) return;
    const real blend = 1.0f - std::exp(-dt / cfg_.smoothing);
    if (!started_) {
      started_ = true;
      neutral_ = roll;
    }
    neutral_ += (roll - neutral_) * std::min(real{1}, dt / cfg_.neutral_seconds);
    const real turn = std::clamp(cfg_.turn_per_roll * (roll - neutral_), -cfg_.max_turn, cfg_.max_turn);
    turn_ += (turn - turn_) * blend;
    closed_ += (std::clamp(tilt, real{0}, cfg_.max_closed) - closed_) * blend;
    roll_ += (roll - roll_) * blend;
  }
  void reset() { *this = BatFace{cfg_}; }

  real turn() const { return turn_; }      // + = facing right
  real closed() const { return closed_; }  // 0 = as held at rest
  real roll() const { return roll_; }      // the hand's roll, smoothed, for drawing

 private:
  BatFaceConfig cfg_;
  real turn_ = 0, closed_ = 0, roll_ = 0, neutral_ = 0;
  bool started_ = false;
};

}  // namespace pp
