#include "math/vec3.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <type_traits>

using pp::real;
using pp::Vec3;
using Catch::Matchers::WithinAbs;

static_assert(std::is_trivially_copyable_v<Vec3>);
static_assert(std::is_aggregate_v<Vec3>);
static_assert(sizeof(Vec3) == 3 * sizeof(real));

TEST_CASE("vec3 arithmetic is constexpr and correct", "[vec3]") {
  constexpr Vec3 a{1, 2, 3};
  constexpr Vec3 b{-4, 5, 0.5f};

  constexpr Vec3 sum = a + b;
  constexpr Vec3 diff = a - b;
  constexpr Vec3 neg = -a;
  constexpr Vec3 scaled_r = a * real{2};
  constexpr Vec3 scaled_l = real{2} * a;
  constexpr Vec3 divided = a / real{2};

  CHECK(sum == Vec3{-3, 7, 3.5f});
  CHECK(diff == Vec3{5, -3, 2.5f});
  CHECK(neg == Vec3{-1, -2, -3});
  CHECK(scaled_r == Vec3{2, 4, 6});
  CHECK(scaled_l == scaled_r);
  CHECK(divided == Vec3{0.5f, 1, 1.5f});
}

TEST_CASE("dot and cross", "[vec3]") {
  constexpr Vec3 x{1, 0, 0}, y{0, 1, 0}, z{0, 0, 1};

  constexpr real d = dot(Vec3{1, 2, 3}, Vec3{4, -5, 6});
  CHECK(d == real{12});

  // Right-handed basis.
  constexpr Vec3 xy = cross(x, y);
  constexpr Vec3 yz = cross(y, z);
  constexpr Vec3 zx = cross(z, x);
  CHECK(xy == z);
  CHECK(yz == x);
  CHECK(zx == y);

  constexpr Vec3 a{1.5f, -2, 0.25f}, b{3, 0.5f, -4};
  constexpr Vec3 c = cross(a, b);
  CHECK(c == -cross(b, a));
  CHECK_THAT(dot(c, a), WithinAbs(0.0, 1e-5));
  CHECK_THAT(dot(c, b), WithinAbs(0.0, 1e-5));
}

TEST_CASE("length and normalisation", "[vec3]") {
  constexpr real lsq = length_sq(Vec3{3, 4, 12});
  CHECK(lsq == real{169});
  CHECK_THAT(length(Vec3{3, 4, 12}), WithinAbs(13.0, 1e-5));

  const Vec3 n = normalized(Vec3{0, -7, 0});
  CHECK_THAT(n.x, WithinAbs(0.0, 1e-6));
  CHECK_THAT(n.y, WithinAbs(-1.0, 1e-6));
  CHECK_THAT(n.z, WithinAbs(0.0, 1e-6));
  CHECK_THAT(length(normalized(Vec3{0.3f, -2, 11})), WithinAbs(1.0, 1e-5));

  // Zero in, zero out. No NaN.
  const Vec3 zero = normalized(Vec3{});
  CHECK(zero == Vec3{});
  CHECK_FALSE(std::isnan(zero.x));
}
