#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace pp;
using Catch::Matchers::WithinAbs;

namespace {

// Plays the world forward and returns the first two things the ball touches.
struct Touches {
  ContactMask first = contact::kNone, second = contact::kNone;
  Vec3 first_pos, second_pos;
};

Touches play(World w, BallState ball) {
  w.ball = ball;
  Touches t;
  for (int i = 0; i < 3000; ++i) {
    const ContactMask c = step(w, {}, 1e-3f);
    if (c == contact::kNone) continue;
    if (t.first == contact::kNone) {
      t.first = c;
      t.first_pos = w.ball.pos;
    } else {
      t.second = c;
      t.second_pos = w.ball.pos;
      break;
    }
  }
  return t;
}

}  // namespace

TEST_CASE("plan_shot lands on target over the net, from either end", "[shot]") {
  const World w;
  const real side = GENERATE(1.0f, -1.0f);  // who is hitting
  const real speed = GENERATE(4.0f, 7.0f, 10.0f);
  const real from_y = GENERATE(0.80f, 1.00f, 1.25f);
  const real target_x = GENERATE(-0.5f, 0.0f, 0.55f);
  const Vec3 spin{-60.0f * side, 0, 0};  // topspin for that direction of travel

  const Vec3 from{0.3f, from_y, side * 1.5f};
  const Vec3 target{target_x, w.table.height + w.ball_params.radius, -side * 0.9f};
  const auto v = plan_shot(w, from, spin, target, speed);
  REQUIRE(v.has_value());

  // Judge it with the real collision code, not the planner's own flight model.
  const Touches t = play(w, {.pos = from, .vel = *v, .spin = spin});
  const ContactMask far_half = side > 0 ? contact::kTableFar : contact::kTableNear;
  REQUIRE(t.first == far_half);
  CHECK_THAT(t.first_pos.x, WithinAbs(static_cast<double>(target.x), 0.03));
  // On target, or deeper when the planner had to lengthen a soft ball.
  CHECK(std::abs(t.first_pos.z) > std::abs(target.z) - 0.03f);
  CHECK(std::abs(t.first_pos.z) < w.table.length * 0.5f);
  // It must carry off the end: no second bounce on the table.
  CHECK((t.second & (contact::kTableNear | contact::kTableFar)) == 0);
}

TEST_CASE("plan_serve produces a legal serve from a range of contact points", "[shot]") {
  const World w;
  const real side = GENERATE(1.0f, -1.0f);
  const real height = GENERATE(0.10f, 0.20f, 0.30f, 0.42f);  // above the table
  const real from_x = GENERATE(-0.5f, 0.0f, 0.4f);
  const real pace = GENERATE(0.0f, 0.5f, 1.0f);

  const Vec3 from{from_x, w.table.height + height, side * 1.42f};
  const auto serve = plan_serve(w, from, -from_x, pace);
  REQUIRE(serve.has_value());

  const Touches t = play(w, {.pos = from, .vel = serve->vel, .spin = serve->spin});
  const ContactMask own = side > 0 ? contact::kTableNear : contact::kTableFar;
  const ContactMask far_half = side > 0 ? contact::kTableFar : contact::kTableNear;
  CHECK(t.first == own);
  CHECK(t.second == far_half);
  CHECK(std::abs(t.second_pos.z) < w.table.length * 0.5f);
}

TEST_CASE("a faster requested serve really is faster", "[shot]") {
  const World w;
  const Vec3 from{0, w.table.height + 0.25f, 1.42f};
  const auto slow = plan_serve(w, from, 0, 0.0f);
  const auto fast = plan_serve(w, from, 0, 1.0f);
  REQUIRE((slow && fast));
  CHECK(length(fast->vel) > length(slow->vel));
}

TEST_CASE("predict_crossing agrees with what the simulation then does", "[shot]") {
  World w;
  w.ball = {.pos = {0.1f, 1.05f, -1.3f}, .vel = {0.3f, 1.6f, 4.4f}, .spin = {60, 0, 0}};
  const real plane = 1.5f;

  const auto c = predict_crossing(w, {0, 0, plane}, {0, 0, -1});
  REQUIRE(c.has_value());
  CHECK(c->bounced);

  Vec3 prev = w.ball.pos;
  real t = 0;
  for (int i = 0; i < 3000 && w.ball.pos.z < plane; ++i) {
    prev = w.ball.pos;
    step(w, {}, 1e-3f);
    t += 1e-3f;
  }
  REQUIRE(w.ball.pos.z >= plane);
  // The planner steps at 2 ms and the game at 1 ms, so allow a centimetre.
  CHECK_THAT(c->point.x, WithinAbs(static_cast<double>(w.ball.pos.x), 0.01));
  CHECK_THAT(c->point.y, WithinAbs(static_cast<double>(w.ball.pos.y), 0.015));
  CHECK_THAT(c->time, WithinAbs(static_cast<double>(t), 0.01));
}

TEST_CASE("predict_crossing works against a tilted plane", "[shot]") {
  World w;
  w.ball = {.pos = {0.1f, 1.0f, -1.3f}, .vel = {0.3f, 1.3f, 6.5f}, .spin = {40, 0, 0}};
  const Vec3 point{0, 0.96f, 1.50f};
  const Vec3 toward_ball = normalized(Vec3{0, -0.5f, -1});

  const auto c = predict_crossing(w, point, toward_ball);
  REQUIRE(c.has_value());
  CHECK_THAT(dot(c->point - point, toward_ball), WithinAbs(0.0, 1e-4));  // lies on the plane
  CHECK(c->bounced);
}

TEST_CASE("predict_crossing reports a ball that never arrives", "[shot]") {
  World w;
  w.ball = {.pos = {0, 0.9f, -1.0f}, .vel = {0, 0, 2.0f}, .spin = {}};  // dies in the net
  CHECK_FALSE(predict_crossing(w, {0, 0, 1.5f}, {0, 0, -1}).has_value());
}
