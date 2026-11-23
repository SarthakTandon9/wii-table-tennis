#pragma once
// From noisy, late palm measurements to a paddle pose in the game world, in
// three dimensions, valid for "now".
//
//   x, y   where the palm is in the image
//   depth  from how big the hand looks. A pinhole camera makes an object's
//          image size inversely proportional to its distance, so the ratio of
//          the hand's current size to its resting size gives how far it has
//          moved toward or away from the camera. One camera cannot do better
//          than that, and it is noisy, so depth is filtered harder and given
//          a limited range.
//
// Each axis runs its own Kalman filter in the camera's time base. Asking for
// the pose "now" extrapolates from the last frame by its age, which is what
// cancels the camera's delay (about 70 ms here).

#include "core/types.hpp"
#include "estimation/kalman.hpp"
#include "estimation/palm.hpp"
#include "math/vec3.hpp"
#include "platform/hand_tracker.hpp"

namespace pp {

struct HandPose {
  Vec3 pos;        // world, metres
  Vec3 vel;        // world, metres per second of REAL time
  // The hand's own speed, before any mapping to the game: x and y in image
  // widths per second, z as how fast the hand is pushing toward the camera
  // (rate of growth of its apparent size, never negative). Swing
  // detection uses this, so it does not change when the mapping is retuned.
  //
  // This is NOT the Kalman filter's velocity. It is plain distance travelled
  // over the last tenth of a second. The filter's velocity is built to react
  // instantly to surprise, which is right for placing the paddle and wrong
  // for judging a swing: replaying recorded play, one bad camera frame made
  // it report speeds no hand can reach, and it would have fired about 23
  // false swings a minute.
  Vec3 image_vel;
  // The same, over the newest single frame interval (33 ms) instead of two.
  // It knows about a swing one camera frame sooner and is twice as noisy.
  // Used only while a ball is about to arrive, when sooner is worth it.
  Vec3 image_vel_fast;
  real roll = 0;   // radians, 0 = fingers up, positive = fingers tilted to the right
  // How far the palm is tipped out of the picture plane, in radians, 0 = flat
  // toward the camera. Read from foreshortening: a palm tipped forward or
  // back looks shorter (wrist to knuckles) while staying as wide. One camera
  // cannot tell forward from back (both shorten it alike), so this is a
  // size, not a direction; the game takes it as tipped forward, over the
  // ball, which is how a hand closes on a stroke.
  real tilt = 0;
  bool valid = false;
};

struct PalmFilterConfig {
  // Where in the image the hand plays. By default this is LEARNED: wherever
  // the hand is when it first appears becomes the paddle's rest position.
  // (A fixed region in the middle of the frame was tried first. Recorded play
  // showed the hand actually sat along the bottom edge of that region, so the
  // paddle was pinned low and 14 cm off to one side.)
  bool learn_centre = true;
  float centre_x = 0.5f, centre_y = 0.5f;   // used as given when not learning
  // How much hand travel, as a fraction of the image, spans the whole paddle
  // range. From recorded play: the hand covered about 0.31 of the width.
  float span_x = 0.34f, span_y = 0.26f;
  float rest_v = 0.10f;  // where in the paddle's height range the rest position sits
  // The rest position also follows wherever the hand tends to be, slowly.
  // People drift: in recorded play the hand settled well away from where it
  // first appeared and the bat sat against one edge of its range.
  float centre_follow_seconds = 6.0f;  // 0 = never move it after the first frames
  // Sideways, the mapping fits itself to how the player actually moves.
  // The goal is that a ROUGH movement of the hand moves the bat: hand to the
  // left, bat to the left, without the player having to find out how far is
  // far enough. So the filter keeps the range the resting hand has lately
  // covered (it widens at once, narrows slowly) and stretches that range over
  // the table. Someone who moves the hand 10 cm and someone who moves it 30
  // both reach both sides. Recorded play is why: with a fixed scale the bat
  // sat pinned against one edge of its range for 24 of 86 balls.
  // Only a calm hand counts; a swing must not stretch the range.
  // Off when centre_follow_seconds is 0 (the mapping is then fixed).
  float range_min_x = 0.18f;            // never more sensitive than this (image widths, edge to edge)
  float range_widen_seconds = 0.25f;
  float range_narrow_seconds = 25.0f;
  float range_calm_speed = 0.6f;        // image widths per second
  // The box in the world that maps to.
  real world_half_width = 0.75f;
  // The lowest the bat goes is about where balls arrive (0.90). Recorded play
  // had the hand resting at the very bottom of the picture, which pinned the
  // bat at the old minimum of 0.70, a constant 17-20 cm under every ball.
  real world_min_y = 0.86f, world_max_y = 1.36f;
  real base_z = 1.50f;            // paddle depth with the hand at rest
  // Depth moves the bat only a little, for show. One camera cannot measure
  // distance well enough to play on: in recorded sessions the hand's apparent
  // size varied by a factor of two as it turned and closed, which the first
  // version turned into 64 cm of paddle travel and a ball that could not be met.
  real depth_gain = 0.25f;        // metres of paddle travel per unit of ln(size ratio)
  real max_forward = 0.10f, max_back = 0.05f;

  // Measurement noise (standard deviation), in image units and in ln(size).
  float pos_noise = 0.008f;  // measured from recorded play: 5.3 px of a 640-wide image
  float depth_noise = 0.030f;
  // Process noise: the hand acceleration the filter assumes while things go
  // as predicted, and how sharply that is raised when a measurement lands
  // far from the prediction (see observe()).
  // Chosen by sweeping both against four measures at once: shake of a still
  // hand, lag at constant speed, frames to catch a sudden swing, and
  // overshoot when the hand stops dead.
  float pos_accel = 0.3f, pos_surprise = 25.0f;
  float depth_accel = 0.2f, depth_surprise = 10.0f;

  // Below `still_speed` (image widths per second) the hand counts as at rest
  // and the pose is not extrapolated at all; by `moving_speed` it is fully.
  // Predicting ahead only pays when there is motion to predict. For a resting
  // hand it would just turn velocity noise into shake.
  float still_speed = 0.03f, moving_speed = 0.12f;

  // A frame that puts the hand further than this from where it was expected
  // (fraction of the image, in one frame) is a detection glitch, not a
  // movement: the fastest real swing covers about a third of that. Such a
  // frame is ignored. If the hand really is there, the next frames agree and
  // tracking restarts from the new place.
  float glitch_jump = 0.30f;      // per frame: 9 image widths a second, three times a hard swing
  int glitch_patience = 2;        // frames ignored before believing the jump

  // Never extrapolate further than this. It was 0.11 s (the whole camera
  // delay) while the player had to place the bat exactly. Now that placing is
  // rough and the magnet finishes it, most of that is not worth its price:
  // extrapolating multiplies velocity noise, and on recorded frames it was a
  // large share of the bat's visible shake.
  double max_lookahead = 0.04;
  double lost_after = 0.30;       // s without a usable palm
};

class PalmFilter {
 public:
  explicit PalmFilter(PalmFilterConfig config = {}) : cfg_(config) {}

  // Feed every camera observation, in order.
  void observe(const HandObservation& obs);

  // The pose at host-clock time `now` (extrapolated from the last frame).
  HandPose pose_at(double now) const;

  void reset() { *this = PalmFilter{cfg_}; }
  const PalmFilterConfig& config() const { return cfg_; }
  // The sideways mapping as it stands now (it fits itself to the player):
  // the image x that maps to the middle, and the travel that spans the range.
  float centre_x() const { return centre_x_; }
  float span_x() const { return span_x_; }

 private:
  PalmFilterConfig cfg_;
  KalmanCV x_, y_, depth_;   // depth_ tracks ln(apparent hand size)
  float roll_ = 0;
  float tilt_ = 0;
  float flat_ratio_ = 1.35f;  // palm length over width when flat to the camera, as learned
  float rest_log_size_ = 0;  // ln(size) with the hand at its resting distance
  float centre_x_ = 0.5f, centre_y_ = 0.5f;  // image position that maps to the rest position
  float span_x_ = 0.34f;                     // hand travel that covers the table, as learned
  float low_x_ = 0, high_x_ = 0;             // the range the calm hand has lately covered
  int rest_samples_ = 0;
  double last_capture_ = 0;  // time of the last frame folded in
  bool tracking_ = false;
  int glitches_ = 0;         // consecutive frames rejected as glitches

  // The last few accepted measurements, for the travel-based speed.
  struct Sample {
    double time = 0;
    float x = 0, y = 0, log_length = 0, log_width = 0;
  };
  static constexpr int kHistory = 3;  // newest and the one two frames back: about 67 ms
  Sample history_[kHistory];
  int history_count_ = 0;
  void remember(double time, float x, float y, float log_length, float log_width);

  // One frame's position measurement, from a whole palm or a partial one.
  struct Measurement {
    float x = 0, y = 0, confidence = 0, roll = 0;
    bool complete = false;
  };
  bool measure(const HandObservation& obs, Measurement& out);
  float offset_x_[5] = {}, offset_y_[5] = {};  // each rigid landmark's usual offset from the centre
  bool offsets_known_ = false;
  float last_x_ = 0, last_y_ = 0, last_log_length_ = 0, last_log_width_ = 0;
};

// Apparent hand size, robust to the hand tilting. Tilting forward shortens
// the palm's length in the image; turning sideways shortens its width; rarely
// both at once. Taking the larger of the two (width rescaled to match) keeps
// the size, and so the depth, steadier than either alone.
float apparent_hand_size(const HandObservation& obs);

}  // namespace pp
