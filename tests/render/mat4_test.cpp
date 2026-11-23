#include "render/mat4.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace pp;
using namespace pp::render;
using Catch::Matchers::WithinAbs;

namespace {

// Applies a matrix to a point and does the perspective divide.
Vec3 project(const Mat4& m, Vec3 p) {
  float out[4];
  for (int row = 0; row < 4; ++row) {
    out[row] = m.m[0][row] * p.x + m.m[1][row] * p.y + m.m[2][row] * p.z + m.m[3][row];
  }
  return {out[0] / out[3], out[1] / out[3], out[2] / out[3]};
}

}  // namespace

TEST_CASE("the camera matrix puts what it looks at in the middle of the picture", "[render]") {
  const Vec3 eye{0, 1.55f, 3.25f}, target{0, 0.82f, 0};
  const Mat4 view_proj = perspective(0.785f, 16.0f / 9.0f, 0.05f, 60.0f) * look_at(eye, target, {0, 1, 0});
  const Vec3 centre = project(view_proj, target);
  CHECK_THAT(centre.x, WithinAbs(0.0, 1e-4));
  CHECK_THAT(centre.y, WithinAbs(0.0, 1e-4));
  // Metal's depth runs 0 (near) to 1 (far), not -1 to 1.
  CHECK(centre.z > 0.0f);
  CHECK(centre.z < 1.0f);
  // Right is right and up is up.
  CHECK(project(view_proj, target + Vec3{0.5f, 0, 0}).x > 0.0f);
  CHECK(project(view_proj, target + Vec3{0, 0.5f, 0}).y > 0.0f);
  // Nearer things have smaller depth.
  CHECK(project(view_proj, {0, 0.8f, 1.0f}).z < project(view_proj, {0, 0.8f, -1.0f}).z);
}

TEST_CASE("along() lays the unit cylinder between two points", "[render]") {
  const Vec3 a{0.2f, 0.1f, -0.3f}, b{0.5f, 0.9f, 0.4f};
  const Mat4 m = along(a, b, 0.05f);
  CHECK(length(project(m, {0, 0, 0}) - a) < 1e-5f);  // the cylinder's base
  CHECK(length(project(m, {0, 1, 0}) - b) < 1e-5f);  // and its top
  // A point on the rim is 5 cm from the axis.
  const Vec3 rim = project(m, {1, 0, 0});
  CHECK_THAT(length(rim - a), WithinAbs(0.05, 1e-5));
  CHECK_THAT(dot(rim - a, normalized(b - a)), WithinAbs(0.0, 1e-5));
}
