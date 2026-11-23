#pragma once
// From 21 hand landmarks to one palm measurement.
//
// This is the raw, per-frame measurement. It is noisy and arrives late; the
// Kalman filter in M4 is what turns a stream of these into a paddle that
// feels attached to the hand.

#include "platform/hand_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

struct Palm {
  float x = 0, y = 0;    // palm centre, normalised image coordinates (mirrored, y up)
  float size = 0;        // wrist to middle knuckle, as a fraction of image height
  float roll = 0;        // radians; 0 = fingers straight up, positive = tilted to the right
  float confidence = 0;  // weakest of the landmarks used
  bool valid = false;
};

// The palm centre is the average of the wrist and the four finger knuckles
// (MCP joints). Those five points move rigidly with the palm; fingertips do
// not, so including them would make the paddle twitch when fingers curl.
inline Palm palm_from(const HandObservation& obs, float min_confidence = 0.3f) {
  Palm palm;
  if (!obs.found) return palm;

  constexpr HandJoint kRigid[] = {HandJoint::Wrist, HandJoint::IndexMCP, HandJoint::MiddleMCP,
                                  HandJoint::RingMCP, HandJoint::LittleMCP};
  float sum_x = 0, sum_y = 0, weakest = 1;
  for (const HandJoint j : kRigid) {
    const JointPoint& p = obs[j];
    sum_x += p.x;
    sum_y += p.y;
    weakest = std::min(weakest, p.confidence);
  }
  palm.confidence = weakest;
  if (weakest < min_confidence) return palm;

  palm.x = sum_x / 5.0f;
  palm.y = sum_y / 5.0f;

  // x and y are each normalised by their own image dimension, so a distance
  // needs x scaled back by the aspect ratio before the two can be combined.
  const JointPoint& wrist = obs[HandJoint::Wrist];
  const JointPoint& knuckle = obs[HandJoint::MiddleMCP];
  const float dx = (knuckle.x - wrist.x) * obs.aspect;
  const float dy = knuckle.y - wrist.y;
  palm.size = std::hypot(dx, dy);
  palm.roll = std::atan2(dx, dy);
  palm.valid = palm.size > 1e-3f;
  return palm;
}

}  // namespace pp
