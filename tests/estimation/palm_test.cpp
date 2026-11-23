#include "estimation/palm.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numbers>

using namespace pp;
using Catch::Matchers::WithinAbs;

namespace {

// An upright open hand centred at (cx, cy): wrist below, knuckles in a row above.
HandObservation upright_hand(float cx, float cy, float aspect = 16.0f / 9.0f) {
  HandObservation obs;
  obs.found = true;
  obs.aspect = aspect;
  for (JointPoint& j : obs.joints) j = {.x = cx, .y = cy, .confidence = 0.9f};
  auto set = [&](HandJoint j, float dx, float dy) {
    obs.joints[static_cast<std::size_t>(j)] = {.x = cx + dx, .y = cy + dy, .confidence = 0.9f};
  };
  set(HandJoint::Wrist, 0.0f, -0.08f);
  set(HandJoint::IndexMCP, -0.03f, 0.02f);
  set(HandJoint::MiddleMCP, -0.01f, 0.02f);
  set(HandJoint::RingMCP, 0.01f, 0.02f);
  set(HandJoint::LittleMCP, 0.03f, 0.02f);
  return obs;
}

}  // namespace

TEST_CASE("no hand, no palm", "[palm]") {
  HandObservation obs;
  CHECK_FALSE(palm_from(obs).valid);
}

TEST_CASE("the palm centre is the average of wrist and knuckles", "[palm]") {
  const Palm p = palm_from(upright_hand(0.4f, 0.6f));
  REQUIRE(p.valid);
  CHECK_THAT(p.x, WithinAbs(0.4, 1e-5));
  CHECK_THAT(p.y, WithinAbs(0.6, 1e-5));  // (-0.08 + 4 * 0.02) / 5 = 0
  CHECK_THAT(p.confidence, WithinAbs(0.9, 1e-6));
}

TEST_CASE("fingertips do not move the palm", "[palm]") {
  HandObservation open = upright_hand(0.5f, 0.5f);
  HandObservation fist = open;
  for (const HandJoint j : {HandJoint::IndexTip, HandJoint::MiddleTip, HandJoint::RingTip,
                            HandJoint::LittleTip, HandJoint::ThumbTip}) {
    fist.joints[static_cast<std::size_t>(j)].y -= 0.15f;  // curl the fingers down
  }
  const Palm a = palm_from(open), b = palm_from(fist);
  CHECK(a.x == b.x);
  CHECK(a.y == b.y);
}

TEST_CASE("one unsure landmark invalidates the measurement", "[palm]") {
  HandObservation obs = upright_hand(0.5f, 0.5f);
  obs.joints[static_cast<std::size_t>(HandJoint::RingMCP)].confidence = 0.1f;
  const Palm p = palm_from(obs);
  CHECK_FALSE(p.valid);
  CHECK_THAT(p.confidence, WithinAbs(0.1, 1e-6));
}

TEST_CASE("size and roll account for the image aspect ratio", "[palm]") {
  // Upright: wrist to middle knuckle is almost purely vertical.
  const Palm up = palm_from(upright_hand(0.5f, 0.5f));
  CHECK_THAT(up.roll, WithinAbs(0.0, 0.2));
  CHECK(up.size > 0.09f);

  // Rotate the hand a quarter turn to the right: the knuckle is now beside
  // the wrist. In a 2:1 image, 0.05 of the width is 0.10 of the height.
  HandObservation side = upright_hand(0.5f, 0.5f, 2.0f);
  side.joints[static_cast<std::size_t>(HandJoint::Wrist)] = {0.45f, 0.5f, 0.9f};
  side.joints[static_cast<std::size_t>(HandJoint::MiddleMCP)] = {0.50f, 0.5f, 0.9f};
  const Palm p = palm_from(side);
  REQUIRE(p.valid);
  CHECK_THAT(p.size, WithinAbs(0.10, 1e-5));
  CHECK_THAT(p.roll, WithinAbs(std::numbers::pi / 2, 1e-4));
}
