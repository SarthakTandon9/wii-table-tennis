#include "game/bat_face.hpp"

#include "estimation/palm_filter.hpp"
#include "game/opponent_planner.hpp"
#include "game/paddle_surface.hpp"
#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace pp;

namespace {
constexpr real kDt = 1e-3f;
}

TEST_CASE("the face turns with the hand's lean, measured from its usual lean", "[face]") {
  BatFace face;
  // A hand that rests leaning 15 degrees to the right.
  const real rest = 0.26f;
  for (int i = 0; i < 3000; ++i) face.update(rest, 0, false, kDt);
  CHECK(std::abs(face.turn()) < 0.02f);  // its resting lean is "straight ahead"

  // Lean 25 degrees further right: the face turns right.
  for (int i = 0; i < 400; ++i) face.update(rest + 0.44f, 0, false, kDt);
  CHECK(face.turn() > 0.12f);
  CHECK(face_normal(face.turn(), face.closed()).x > 0.10f);  // facing right: the ball goes right

  // And left of its rest: the face turns left.
  for (int i = 0; i < 400; ++i) face.update(rest - 0.44f, 0, false, kDt);
  CHECK(face.turn() < -0.12f);
}

TEST_CASE("a tipped palm closes the face; a swing does not disturb it", "[face]") {
  BatFace face;
  for (int i = 0; i < 500; ++i) face.update(0, 0, false, kDt);
  const real open_y = face_normal(face.turn(), face.closed()).y;
  for (int i = 0; i < 500; ++i) face.update(0, 0.5f, false, kDt);
  CHECK(face.closed() > 0.45f);
  CHECK(face_normal(face.turn(), face.closed()).y < open_y - 0.3f);  // tipped over the ball

  // During a swing the readings go wild; the face holds.
  const real before = face.closed();
  for (int i = 0; i < 200; ++i) face.update(1.2f, 0, true, kDt);
  CHECK(face.closed() == before);
  CHECK(std::abs(face.turn()) < 0.02f);
}

TEST_CASE("the filter reads tilt from a foreshortened palm", "[face][filter]") {
  // Wrist below the knuckle row; `length` is wrist to middle knuckle.
  const auto hand = [](double t, float length) {
    HandObservation obs;
    obs.found = true;
    obs.aspect = 4.0f / 3.0f;
    obs.capture_time = t;
    for (JointPoint& j : obs.joints) j = {.x = 0.5f, .y = 0.5f, .confidence = 0.9f};
    obs.joints[static_cast<std::size_t>(HandJoint::Wrist)].y = 0.5f - length;
    obs.joints[static_cast<std::size_t>(HandJoint::IndexMCP)].x = 0.5f - 0.03f;
    obs.joints[static_cast<std::size_t>(HandJoint::RingMCP)].x = 0.5f + 0.015f;
    obs.joints[static_cast<std::size_t>(HandJoint::LittleMCP)].x = 0.5f + 0.03f;
    return obs;
  };
  PalmFilter filter;
  double t = 0;
  for (int f = 0; f < 600; ++f, t += 1.0 / 30) filter.observe(hand(t, 0.10f));  // flat, for 20 s
  CHECK(filter.pose_at(t).tilt < 0.05f);
  // Tipped 45 degrees: the palm looks 0.71 times as long, and no narrower.
  for (int f = 0; f < 20; ++f, t += 1.0 / 30) filter.observe(hand(t, 0.071f));
  const real tilt = filter.pose_at(t).tilt;
  CHECK(tilt > 0.45f);  // about 0.6 rad expected; the dead zone and smoothing take a little
  CHECK(tilt < 0.85f);
}

TEST_CASE("the opponent's planner plays legal shots, away from the bat", "[planner][opponent]") {
  World world;
  world.ball.pos = {0.1f, 0.90f, -1.58f};
  OpponentPlanner planner;
  std::uint32_t rng = 0xFEEDu;
  const OpponentSkill expert = kOpponentSkill[kDifficultyLevels - 1];

  int to_the_left = 0, shots = 0;
  for (int i = 0; i < 60; ++i) {
    const auto shot = planner.choose(world, expert, /*player_bat_x=*/0.5f, rng);
    REQUIRE(shot);
    CHECK(shot->playable > 10);
    // What the batch promised, the full simulation delivers: over the net,
    // onto the player's half, and on past the end line.
    const Flight flight = fly(world, {.pos = world.ball.pos, .vel = shot->vel, .spin = shot->spin});
    CHECK(flight.landed);
    CHECK(flight.crossed_net);
    CHECK(flight.net_clearance > 0.02f);
    CHECK(std::abs(flight.landing.x) < world.table.width * 0.5f);
    CHECK(flight.landing.z > 0.0f);
    CHECK(flight.landing.z < world.table.length * 0.5f);
    CHECK(length(flight.landing - Vec3{shot->landing.x, flight.landing.y, shot->landing.z}) < 0.005f);
    CHECK(stroke_carries(world, {.pos = world.ball.pos, .vel = shot->vel, .spin = shot->spin}));
    ++shots;
    to_the_left += flight.landing.x < 0;
  }
  // The bat is on the right, so a good player goes left nearly every time.
  CHECK(to_the_left > shots * 3 / 4);
}
