#include "estimation/palm_filter.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

namespace {

constexpr float kPalmLengthOverWidth = 1.35f;  // wrist-to-knuckle vs knuckle row, typical hand
constexpr int kRestSamples = 12;               // frames averaged to learn the resting size

float span(const HandObservation& obs, HandJoint a, HandJoint b) {
  return std::hypot((obs[a].x - obs[b].x) * obs.aspect, obs[a].y - obs[b].y);
}

// One predict + update, with the filter's stiffness chosen by how surprising
// the measurement is.
//
// The tension: to cancel the camera delay the pose is extrapolated about
// 70-100 ms ahead using the velocity estimate, so any noise in that velocity
// is multiplied into visible shake. A steady velocity needs a stiff filter
// (small assumed acceleration). But a stiff filter is slow to notice a swing.
//
// Resolution: stay stiff while measurements land where predicted, and loosen
// only when one lands further away than noise can explain. "Further than
// noise can explain" is measured in standard deviations of the expected
// prediction error (the normalised innovation); beyond about 1.5 of those,
// the hand must have accelerated, and the assumed acceleration is raised for
// this step so the filter catches up at once.
void fold_in(KalmanCV& k, float z, float var, float dt, float accel, float surprise_gain) {
  KalmanCV trial = k;
  trial.predict(dt, accel);
  const float sigmas = std::abs(z - trial.p) / std::sqrt(trial.P00 + var);
  // Capped: beyond a point, more surprise means a bad measurement, not a
  // harder acceleration.
  const float excess = std::clamp(sigmas - 1.5f, 0.0f, 3.0f);
  k.predict(dt, accel * (1.0f + surprise_gain * excess * excess));
  k.update(z, var);
}

float shortest_turn(float from, float to) {
  constexpr float kPi = 3.14159265f;
  float d = to - from;
  while (d > kPi) d -= 2 * kPi;
  while (d < -kPi) d += 2 * kPi;
  return d;
}

}  // namespace

float apparent_hand_size(const HandObservation& obs) {
  const float length = span(obs, HandJoint::Wrist, HandJoint::MiddleMCP);
  const float width = span(obs, HandJoint::IndexMCP, HandJoint::LittleMCP);
  return std::max(length, width * kPalmLengthOverWidth);
}

// The five landmarks that move rigidly with the palm.
constexpr HandJoint kRigid[5] = {HandJoint::Wrist, HandJoint::IndexMCP, HandJoint::MiddleMCP,
                                 HandJoint::RingMCP, HandJoint::LittleMCP};

// Turns one camera frame into a position measurement, if it can.
//
// Normally that is the palm centre from all five rigid landmarks. But in
// recorded play the palm was unusable in 15-30% of frames, and worst exactly
// when it mattered: during a fast swing, or with the wrist cut off by the
// bottom of the picture, one or two landmarks lose confidence. Rather than
// drop those frames, the centre is reconstructed from whichever landmarks are
// still good, using where each one usually sits relative to the centre
// (learned from the complete frames).
bool PalmFilter::measure(const HandObservation& obs, Measurement& out) {
  if (!obs.found) return false;
  const Palm palm = palm_from(obs);
  if (palm.valid) {
    out = {palm.x, palm.y, palm.confidence, palm.roll, true};
    // Remember each landmark's usual offset from the centre.
    for (int k = 0; k < 5; ++k) {
      const JointPoint& j = obs[kRigid[k]];
      const float blend = offsets_known_ ? 0.2f : 1.0f;
      offset_x_[k] += (j.x - palm.x - offset_x_[k]) * blend;
      offset_y_[k] += (j.y - palm.y - offset_y_[k]) * blend;
    }
    offsets_known_ = true;
    return true;
  }
  if (!offsets_known_) return false;

  float sum_x = 0, sum_y = 0;
  int used = 0;
  for (int k = 0; k < 5; ++k) {
    const JointPoint& j = obs[kRigid[k]];
    if (j.confidence < 0.2f) continue;
    sum_x += j.x - offset_x_[k];
    sum_y += j.y - offset_y_[k];
    ++used;
  }
  if (used < 2) return false;
  out = {sum_x / static_cast<float>(used), sum_y / static_cast<float>(used), 0.3f, roll_, false};
  return true;
}

void PalmFilter::observe(const HandObservation& obs) {
  Measurement m;
  if (!measure(obs, m)) return;  // nothing usable: the filters simply coast

  // Two independent measures of how big the hand looks. They are only
  // trusted from a complete palm; otherwise the previous values stand.
  float log_length = last_log_length_, log_width = last_log_width_;
  if (m.complete) {
    const float length = span(obs, HandJoint::Wrist, HandJoint::MiddleMCP);
    const float width = span(obs, HandJoint::IndexMCP, HandJoint::LittleMCP);
    if (length < 1e-3f || width < 1e-3f) return;
    log_length = std::log(length);
    log_width = std::log(width);
  } else if (!tracking_) {
    return;  // cannot start tracking from a partial hand
  }
  const float log_size = std::max(log_length, log_width + std::log(kPalmLengthOverWidth));

  const double gap = obs.capture_time - last_capture_;
  bool restart = !tracking_ || gap > cfg_.lost_after || gap <= 0;
  if (!restart) {
    // Glitch gate: how far did the measurement move, per camera frame, since
    // the last one that was accepted? Measured against the last MEASUREMENT.
    // (The first version compared against the filter's prediction, which
    // lags a fast hand; real swings were rejected as glitches and the speed
    // history wiped, so the hardest swings were the ones never seen.)
    const float frames = std::max(1.0f, static_cast<float>(gap * 30.0));
    const float jump = std::hypot(m.x - last_x_, m.y - last_y_) / frames;
    if (jump > cfg_.glitch_jump) {
      if (++glitches_ <= cfg_.glitch_patience) return;  // ignore it and coast
      restart = true;  // it keeps saying so: the hand really is over there
    } else {
      glitches_ = 0;
    }
  }
  last_x_ = m.x;
  last_y_ = m.y;
  last_log_length_ = log_length;
  last_log_width_ = log_width;

  if (restart) {
    // First sight of the hand (or first after losing it): start from here.
    x_.reset(m.x, cfg_.pos_noise * cfg_.pos_noise, 1.0f);
    y_.reset(m.y, cfg_.pos_noise * cfg_.pos_noise, 1.0f);
    depth_.reset(log_size, cfg_.depth_noise * cfg_.depth_noise, 1.0f);
    roll_ = m.roll;
    tilt_ = 0;
    if (rest_samples_ == 0) {
      rest_log_size_ = log_size;
      centre_x_ = cfg_.learn_centre ? m.x : cfg_.centre_x;
      centre_y_ = cfg_.learn_centre ? m.y : cfg_.centre_y;
      span_x_ = cfg_.span_x;
      low_x_ = centre_x_ - 0.5f * span_x_;
      high_x_ = centre_x_ + 0.5f * span_x_;
    }
    tracking_ = true;
    glitches_ = 0;
    history_count_ = 0;  // a jump is not a movement: do not let it read as speed
    remember(obs.capture_time, m.x, m.y, log_length, log_width);
    last_capture_ = obs.capture_time;
    return;
  }
  remember(obs.capture_time, m.x, m.y, log_length, log_width);
  const float dt = static_cast<float>(gap);
  last_capture_ = obs.capture_time;

  // A landmark the detector is unsure of is a noisier measurement.
  const float trust = std::clamp(m.confidence, 0.3f, 1.0f);
  const float pos_var = cfg_.pos_noise * cfg_.pos_noise / (trust * trust);

  fold_in(x_, m.x, pos_var, dt, cfg_.pos_accel, cfg_.pos_surprise);
  fold_in(y_, m.y, pos_var, dt, cfg_.pos_accel, cfg_.pos_surprise);
  if (m.complete) {
    fold_in(depth_, log_size, cfg_.depth_noise * cfg_.depth_noise / (trust * trust), dt,
            cfg_.depth_accel, cfg_.depth_surprise);
  } else {
    depth_.predict(dt, cfg_.depth_accel);
  }
  roll_ += shortest_turn(roll_, m.roll) * 0.35f;

  // Tilt, from the palm's proportions (see HandPose::tilt). The flat
  // proportion differs from hand to hand, so it is learned, as the MEDIAN
  // of what is seen: nudged up a fixed step when the palm looks longer than
  // the estimate and down when shorter, which settles where half the frames
  // are on each side. (An average was tried first. A palm turned sideways
  // looks narrow, so very long for its width, and those frames dragged the
  // average up until an ordinary flat palm read as tipped 20 degrees.)
  // A palm shorter than that for its width is tipped, by the angle whose
  // cosine is the shortfall. Small shortfalls are noise (the ratio wobbles
  // by 2% from frame to frame, and near zero the arc-cosine turns 2% into 11
  // degrees), so the first 12% is ignored.
  if (m.complete) {
    const float ratio = std::exp(log_length - log_width);
    flat_ratio_ += (ratio > flat_ratio_ ? 1.0f : -1.0f) * 0.06f * dt;
    const float shortfall = std::clamp(ratio / flat_ratio_ + 0.12f, 0.0f, 1.0f);
    tilt_ += (std::acos(shortfall) - tilt_) * 0.30f;
  }

  // Learn the resting size from the first frames, then let it follow slow
  // changes in posture (about half a minute), never fast ones (a reach).
  if (rest_samples_ < kRestSamples && m.complete) {
    // The first frames define "at rest": distance, and position in the image.
    ++rest_samples_;
    const float share = 1.0f / static_cast<float>(rest_samples_);
    rest_log_size_ += (log_size - rest_log_size_) * share;
    if (cfg_.learn_centre) {
      centre_x_ += (m.x - centre_x_) * share;
      centre_y_ += (m.y - centre_y_) * share;
      low_x_ = centre_x_ - 0.5f * span_x_;
      high_x_ = centre_x_ + 0.5f * span_x_;
    }
  } else if (rest_samples_ >= kRestSamples) {
    rest_log_size_ += (depth_.p - rest_log_size_) * std::min(1.0f, dt / 30.0f);
    if (cfg_.learn_centre && cfg_.centre_follow_seconds > 0) {
      const float share = std::min(1.0f, dt / cfg_.centre_follow_seconds);
      centre_y_ += (y_.p - centre_y_) * share;
      // Sideways: fit the range to where the calm hand goes.
      if (std::hypot(x_.v, y_.v) < cfg_.range_calm_speed) {
        const float widen = std::min(1.0f, dt / cfg_.range_widen_seconds);
        const float narrow = std::min(1.0f, dt / cfg_.range_narrow_seconds);
        low_x_ += (x_.p - low_x_) * (x_.p < low_x_ ? widen : narrow);
        high_x_ += (x_.p - high_x_) * (x_.p > high_x_ ? widen : narrow);
        if (high_x_ - low_x_ < cfg_.range_min_x) {
          const float mid = 0.5f * (low_x_ + high_x_);
          low_x_ = mid - 0.5f * cfg_.range_min_x;
          high_x_ = mid + 0.5f * cfg_.range_min_x;
        }
        centre_x_ = 0.5f * (low_x_ + high_x_);
        span_x_ = high_x_ - low_x_;
      }
    }
  }
}

void PalmFilter::remember(double time, float x, float y, float log_length, float log_width) {
  if (history_count_ == kHistory) {
    for (int i = 1; i < kHistory; ++i) history_[i - 1] = history_[i];
    --history_count_;
  }
  history_[history_count_++] = {time, x, y, log_length, log_width};
}

HandPose PalmFilter::pose_at(double now) const {
  HandPose pose;
  const double age = now - last_capture_;
  if (!tracking_ || age > cfg_.lost_after) return pose;

  // Extrapolate from the last frame to now. Capped: velocity is an estimate,
  // and multiplying it by a long time multiplies its error too.
  const float speed = std::hypot(x_.v, y_.v);
  const float moving =
      std::clamp((speed - cfg_.still_speed) / (cfg_.moving_speed - cfg_.still_speed), 0.0f, 1.0f);
  const float ahead = static_cast<float>(std::clamp(age, 0.0, cfg_.max_lookahead)) *
                      (moving * moving * (3.0f - 2.0f * moving));
  const float u = std::clamp(0.5f + (x_.extrapolate(ahead) - centre_x_) / span_x_, 0.0f, 1.0f);
  const float v =
      std::clamp(cfg_.rest_v + (y_.extrapolate(ahead) - centre_y_) / cfg_.span_y, 0.0f, 1.0f);
  const real height = cfg_.world_max_y - cfg_.world_min_y;

  // Bigger in the image = closer to the camera = reaching toward the table.
  const real reach = std::clamp(cfg_.depth_gain * (depth_.extrapolate(ahead) - rest_log_size_),
                                -cfg_.max_back, cfg_.max_forward);

  pose.pos = {(2.0f * u - 1.0f) * cfg_.world_half_width, cfg_.world_min_y + v * height,
              cfg_.base_z - reach};
  pose.vel = {x_.v / span_x_ * 2.0f * cfg_.world_half_width, y_.v / cfg_.span_y * height,
              -cfg_.depth_gain * depth_.v};
  if (history_count_ == kHistory) {
    const Sample& newest = history_[kHistory - 1];
    const Sample& oldest = history_[0];
    const float span_s = static_cast<float>(newest.time - oldest.time);
    if (span_s > 1e-3f && span_s < 0.25f) {
      // Reaching toward the camera makes the palm look longer AND wider. A
      // hand that merely turns changes one and not the other, and that was
      // the main source of false swings in recorded play. So the push rate
      // is the smaller of the two growth rates, and only growth counts.
      const float longer = (newest.log_length - oldest.log_length) / span_s;
      const float wider = (newest.log_width - oldest.log_width) / span_s;
      pose.image_vel = {(newest.x - oldest.x) / span_s, (newest.y - oldest.y) / span_s,
                        std::max(0.0f, std::min(longer, wider))};
      const Sample& previous = history_[kHistory - 2];
      const float step_s = static_cast<float>(newest.time - previous.time);
      pose.image_vel_fast = pose.image_vel;
      if (step_s > 1e-3f && step_s < 0.12f) {
        pose.image_vel_fast.x = (newest.x - previous.x) / step_s;
        pose.image_vel_fast.y = (newest.y - previous.y) / step_s;
      }
    }
  }
  pose.roll = roll_;
  pose.tilt = tilt_;
  pose.valid = true;
  return pose;
}

}  // namespace pp
