#pragma once
// Taking the shake out of the bat without making it feel slow.
//
// The bat's position comes from a hand seen by a camera: about half a pixel
// of noise per frame, stretched over the whole table (a hand that moves 20%
// of the picture moves the bat 1.5 m, so half a pixel is several
// millimetres), then extrapolated ahead to cancel the camera's delay, which
// multiplies the noise again. In recorded play a hand held nearly still moved
// the bat 2.5 mm per displayed frame at the median and 20 mm one frame in
// ten, each in a different direction. That is the jitter.
//
// A plain low-pass filter trades one fault for another: smooth enough to
// hide the shake, it trails behind every real movement. The "one euro"
// filter (Casiez, Roussel and Vogel, 2012) avoids the trade by changing its
// own cutoff: when the bat is moving slowly it filters hard (shake is
// visible, lag is not), and the faster it moves the less it filters (lag is
// visible, shake is not).
//
// Lag costs nothing here in any case. Whether the ball is hit depends on the
// swing, which is read from the hand's own speed, not from this position;
// and the magnet finishes placing the bat.

#include "core/types.hpp"
#include "math/vec3.hpp"

#include <cmath>
#include <numbers>

namespace pp {

struct BatSmootherConfig {
  real min_cutoff = 1.2f;   // Hz: how hard a slow bat is filtered
  real beta = 1.6f;         // Hz per m/s: how quickly speed opens the filter up
  real speed_cutoff = 2.0f; // Hz: smoothing of the speed estimate itself
};

class BatSmoother {
 public:
  explicit BatSmoother(BatSmootherConfig config = {}) : cfg_(config) {}

  // Call with the unsmoothed position and the real time since the last call.
  Vec3 update(Vec3 raw, real dt) {
    if (!started_ || dt <= 0) {
      started_ = true;
      pos_ = raw;
      vel_ = {};
      last_raw_ = raw;
      return pos_;
    }
    // How fast is it really going? Estimated from the raw input, smoothed.
    const Vec3 raw_vel = (raw - last_raw_) / dt;
    last_raw_ = raw;
    vel_ += (raw_vel - vel_) * blend(cfg_.speed_cutoff, dt);
    // The faster, the higher the cutoff, the closer the output follows.
    const real cutoff = cfg_.min_cutoff + cfg_.beta * length(vel_);
    pos_ += (raw - pos_) * blend(cutoff, dt);
    return pos_;
  }

  void reset() { started_ = false; }
  const BatSmootherConfig& config() const { return cfg_; }

 private:
  // Weight of the new sample for a first-order low-pass at `cutoff` Hz.
  static real blend(real cutoff, real dt) {
    return 1.0f - std::exp(-2.0f * std::numbers::pi_v<real> * cutoff * dt);
  }

  BatSmootherConfig cfg_;
  Vec3 pos_, vel_, last_raw_;
  bool started_ = false;
};

}  // namespace pp
