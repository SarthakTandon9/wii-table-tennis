#pragma once
// Where the player's paddle can be.
//
// The paddle lives on a tilted surface instead of a flat wall: raising it
// also carries it forward, toward the table. An upward flick is therefore a
// forward-and-up stroke, which is how a real drive is played, and it gives
// the ball real forward paddle speed to work with.
//
// This is a stand-in for depth. Once the hand's distance from the camera is
// tracked (M4), the paddle gets a true third axis and this surface goes away.

#include "core/types.hpp"
#include "estimation/palm.hpp"
#include "math/vec3.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

inline constexpr Vec3 kSurfacePoint{0, 0.96f, 1.50f};  // a point on the surface
inline constexpr real kSurfaceSlope = 0.5f;            // metres forward per metre up
inline constexpr real kPaddleMinY = 0.62f, kPaddleMaxY = 1.55f;
inline constexpr real kPaddleMaxX = 1.2f;

constexpr real surface_z(real y) { return kSurfacePoint.z - kSurfaceSlope * (y - kSurfacePoint.y); }

// Unit normal of the surface, pointing at the table (where the ball comes from).
inline Vec3 surface_toward_ball() { return normalized(Vec3{0, -kSurfaceSlope, -1}); }

// Any x, y -> the nearest allowed point on the surface.
inline Vec3 on_surface(real x, real y) {
  const real cy = std::clamp(y, kPaddleMinY, kPaddleMaxY);
  return {std::clamp(x, -kPaddleMaxX, kPaddleMaxX), cy, surface_z(cy)};
}

// Palm position in the image -> a point on the surface. Only the middle half
// of the image is used, so the hand never has to reach the edge of the frame,
// where the camera starts losing fingers.
inline Vec3 palm_on_surface(const Palm& palm) {
  const real u = std::clamp((palm.x - 0.25f) / 0.5f, 0.0f, 1.0f);
  const real v = std::clamp((palm.y - 0.25f) / 0.5f, 0.0f, 1.0f);
  return on_surface(-0.95f + u * 1.9f, 0.66f + v * (1.45f - 0.66f));
}

// The paddle face for a paddle at `pos`: slightly open, and turned toward the
// middle when out wide. The palm's real orientation replaces this in M4.
// The bat's face from the hand's own attitude (see game/bat_face.hpp for how
// the two angles are obtained). `yaw` turns the face to the right when
// positive; `closed` tips it forward over the ball. With both at zero the
// face is slightly open, as a bat is held at rest.
inline Vec3 face_normal(real yaw, real closed) {
  return normalized(Vec3{std::sin(yaw), 0.30f - 1.1f * std::sin(closed), -std::cos(yaw)});
}

inline Vec3 synthetic_face_normal(Vec3 pos) {
  return normalized(Vec3{-0.3f * pos.x, 0.30f, -1.0f});
}

}  // namespace pp
