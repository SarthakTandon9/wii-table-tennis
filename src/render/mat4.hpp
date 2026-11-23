#pragma once
// 4x4 matrices for the renderer. Column-major (m[column][row]), which is the
// layout Metal's shading language expects for float4x4, so a Mat4 can be
// copied into a uniform buffer as it is.
//
// Clip space is Metal's: x and y in [-1, 1], z in [0, 1] (OpenGL's is
// [-1, 1]; using an OpenGL projection here would waste half the depth range
// and clip the near half of the scene).

#include "math/vec3.hpp"

#include <cmath>

namespace pp::render {

struct Mat4 {
  float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
  Mat4 r;
  for (int c = 0; c < 4; ++c) {
    for (int row = 0; row < 4; ++row) {
      r.m[c][row] = a.m[0][row] * b.m[c][0] + a.m[1][row] * b.m[c][1] + a.m[2][row] * b.m[c][2] +
                    a.m[3][row] * b.m[c][3];
    }
  }
  return r;
}

inline Mat4 translation(Vec3 t) {
  Mat4 r;
  r.m[3][0] = t.x;
  r.m[3][1] = t.y;
  r.m[3][2] = t.z;
  return r;
}

inline Mat4 scaling(Vec3 s) {
  Mat4 r;
  r.m[0][0] = s.x;
  r.m[1][1] = s.y;
  r.m[2][2] = s.z;
  return r;
}

// A frame with the given axes (columns) at `origin`. The axes carry the scale.
inline Mat4 basis(Vec3 x, Vec3 y, Vec3 z, Vec3 origin) {
  Mat4 r;
  r.m[0][0] = x.x, r.m[0][1] = x.y, r.m[0][2] = x.z;
  r.m[1][0] = y.x, r.m[1][1] = y.y, r.m[1][2] = y.z;
  r.m[2][0] = z.x, r.m[2][1] = z.y, r.m[2][2] = z.z;
  r.m[3][0] = origin.x, r.m[3][1] = origin.y, r.m[3][2] = origin.z;
  return r;
}

// Any unit vector at right angles to `n`.
inline Vec3 any_perpendicular(Vec3 n) {
  const Vec3 other = std::abs(n.y) < 0.9f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
  return normalized(cross(other, n));
}

// Maps the unit mesh axis +y onto the segment from `a` to `b` with the given
// radius: turns the unit cylinder or capsule into a limb, a handle, a post.
inline Mat4 along(Vec3 a, Vec3 b, float radius) {
  const Vec3 axis = b - a;
  const float len = length(axis);
  const Vec3 y = len > 1e-6f ? axis / len : Vec3{0, 1, 0};
  const Vec3 x = any_perpendicular(y);
  const Vec3 z = cross(x, y);
  return basis(x * radius, y * len, z * radius, a);
}

inline Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
  const Vec3 f = normalized(target - eye);
  const Vec3 s = normalized(cross(f, up));
  const Vec3 u = cross(s, f);
  Mat4 r;
  r.m[0][0] = s.x, r.m[1][0] = s.y, r.m[2][0] = s.z;
  r.m[0][1] = u.x, r.m[1][1] = u.y, r.m[2][1] = u.z;
  r.m[0][2] = -f.x, r.m[1][2] = -f.y, r.m[2][2] = -f.z;
  r.m[3][0] = -dot(s, eye);
  r.m[3][1] = -dot(u, eye);
  r.m[3][2] = dot(f, eye);
  return r;
}

// Right-handed, looking down -z, depth 0 at `near` and 1 at `far`.
inline Mat4 perspective(float fov_y_radians, float aspect, float near, float far) {
  const float f = 1.0f / std::tan(fov_y_radians * 0.5f);
  Mat4 r;
  r.m[0][0] = f / aspect;
  r.m[1][1] = f;
  r.m[2][2] = far / (near - far);
  r.m[2][3] = -1;
  r.m[3][2] = near * far / (near - far);
  r.m[3][3] = 0;
  return r;
}

inline Mat4 orthographic(float half_width, float half_height, float near, float far) {
  Mat4 r;
  r.m[0][0] = 1.0f / half_width;
  r.m[1][1] = 1.0f / half_height;
  r.m[2][2] = 1.0f / (near - far);
  r.m[3][2] = near / (near - far);
  return r;
}

}  // namespace pp::render
