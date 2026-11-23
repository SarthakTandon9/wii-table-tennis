#pragma once

#include "core/types.hpp"

#include <cmath>

namespace pp {

// Plain aggregate: trivially copyable, 12 bytes, no padding. That lets it go
// through memcpy-based queues and convert to/from SoA arrays without ceremony.
//
// Passed by value everywhere. On arm64 a struct of up to four floats is a
// "homogeneous floating-point aggregate" and travels in vector registers
// (v0-v2 here), so by-value is cheaper than a pointer to the stack and it
// tells the optimiser the arguments cannot alias.
struct Vec3 {
  real x{}, y{}, z{};
  friend constexpr bool operator==(const Vec3&, const Vec3&) = default;
};

constexpr Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
constexpr Vec3 operator*(Vec3 a, real s) { return {a.x * s, a.y * s, a.z * s}; }
constexpr Vec3 operator*(real s, Vec3 a) { return a * s; }
constexpr Vec3 operator/(Vec3 a, real s) { return {a.x / s, a.y / s, a.z / s}; }

constexpr Vec3& operator+=(Vec3& a, Vec3 b) { return a = a + b; }
constexpr Vec3& operator-=(Vec3& a, Vec3 b) { return a = a - b; }
constexpr Vec3& operator*=(Vec3& a, real s) { return a = a * s; }

constexpr real dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Right-handed: cross(x, y) == z.
constexpr Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

constexpr real length_sq(Vec3 a) { return dot(a, a); }

// std::sqrt is not constexpr until C++26, so these two are plain inline.
inline real length(Vec3 a) { return std::sqrt(length_sq(a)); }

// Zero in, zero out (never NaN).
inline Vec3 normalized(Vec3 a) {
  const real len = length(a);
  return len > real{0} ? a / len : Vec3{};
}

constexpr Vec3 lerp(Vec3 a, Vec3 b, real t) { return a + (b - a) * t; }

}  // namespace pp
