#include "physics/world.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cstring>

using namespace pp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr real kDt = 1e-3f;

World world_with_ball(Vec3 pos, Vec3 vel, Vec3 spin = {}) {
  World w;
  w.ball = {.pos = pos, .vel = vel, .spin = spin};
  return w;
}

// Runs until `stop` is seen or the time limit passes; returns every contact seen.
ContactMask run(World& w, std::span<const Paddle> paddles, real seconds, real dt = kDt,
                ContactMask stop = contact::kNone) {
  ContactMask seen = contact::kNone;
  const int steps = static_cast<int>(seconds / dt);
  for (int i = 0; i < steps; ++i) {
    seen |= step(w, paddles, dt);
    if (seen & stop) break;
  }
  return seen;
}

}  // namespace

TEST_CASE("a ball dropped on the table rebounds to about e^2 of the drop", "[world][table]") {
  World w = world_with_ball({0, 0, 0.5f}, {});
  w.ball_params.air_density = 0;  // in air, drag alone costs about 5% of the rebound
  const real surface = w.table.height + w.ball_params.radius;
  const real drop = 0.30f;
  w.ball.pos.y = surface + drop;

  real lowest = w.ball.pos.y, apex = 0;
  bool bounced = false;
  for (int i = 0; i < 700; ++i) {
    const ContactMask c = step(w, {}, kDt);
    if (c & contact::kTableNear) bounced = true;
    CHECK_FALSE((c & contact::kTableFar));
    lowest = std::min(lowest, w.ball.pos.y);
    if (bounced) apex = std::max(apex, w.ball.pos.y);
    if (bounced && w.ball.vel.y < 0) break;
  }

  REQUIRE(bounced);
  CHECK(lowest >= surface - 1e-5f);  // never sinks into the table
  const real e = w.table.surface.restitution;
  CHECK_THAT(apex - surface, WithinRel(static_cast<double>(e * e * drop), 0.01));
}

TEST_CASE("the table reports which half was hit", "[world][table]") {
  World far = world_with_ball({0.2f, 1.0f, -0.8f}, {});
  CHECK((run(far, {}, 1.0f, kDt, contact::kTableFar) & contact::kTableFar));
}

TEST_CASE("a ball beside the table falls to the floor", "[world][table]") {
  World w = world_with_ball({1.2f, 1.0f, 0.5f}, {});
  const ContactMask seen = run(w, {}, 1.5f, kDt, contact::kFloor);
  CHECK((seen & contact::kFloor));
  CHECK_FALSE((seen & (contact::kTableNear | contact::kTableFar)));
  CHECK(w.ball.pos.y >= w.ball_params.radius - 1e-5f);
}

TEST_CASE("a smash cannot tunnel through the table, even with a coarse step", "[world][table]") {
  const real dt = GENERATE(1e-3f, 5e-3f, 2e-2f);  // at 20 ms the ball moves 0.8 m per step
  World w = world_with_ball({0, 1.2f, 0.6f}, {0, -40, 0});
  const ContactMask seen = run(w, {}, 0.1f, dt);
  CHECK((seen & contact::kTableNear));
  CHECK(w.ball.pos.y > w.table.height);
  CHECK(w.ball.vel.y > 0);
}

TEST_CASE("a ball coming up from under the table is not bounced by its top", "[world][table]") {
  World w = world_with_ball({0, 0.5f, 0.5f}, {0, 6, 0});
  const ContactMask seen = run(w, {}, 0.05f);
  CHECK(seen == contact::kNone);
}

TEST_CASE("a stationary paddle returns an incoming ball", "[world][paddle]") {
  World w = world_with_ball({0, 1.0f, 1.0f}, {0, 0, 8});
  w.ball_params.air_density = 0;
  w.ball_params.gravity = {};
  const std::array<Paddle, 1> paddle{Paddle{.pos = {0, 1.0f, 1.5f}, .vel = {}, .normal = {0, 0, -1}}};

  const ContactMask seen = run(w, paddle, 0.1f, kDt, contact::kPaddle0);
  REQUIRE((seen & contact::kPaddle0));
  CHECK(w.ball.vel.z < 0);
  CHECK_THAT(-w.ball.vel.z, WithinRel(8.0 * 0.8, 1e-4));  // e = 0.8
  CHECK(w.ball.pos.z < 1.5f);
}

TEST_CASE("a fast swing cannot pass through a hanging ball", "[world][paddle]") {
  // Relative motion is 25 mm to 0.5 m per step against a paddle of zero thickness.
  const real dt = GENERATE(1e-3f, 5e-3f, 2e-2f);
  World w = world_with_ball({0, 1.0f, 1.0f}, {});
  w.ball_params.gravity = {};  // keep the ball still until struck
  w.ball_params.air_density = 0;  // at 45 m/s drag is ~20 g and would blur the check

  Paddle paddle{.pos = {0, 1.0f, 1.6f}, .vel = {0, 0, -25}, .normal = {0, 0, -1}};
  bool struck = false;
  for (int i = 0; i < static_cast<int>(0.06f / dt) && !struck; ++i) {
    struck = (step(w, std::span<const Paddle>{&paddle, 1}, dt) & contact::kPaddle0) != 0;
    paddle.pos += paddle.vel * dt;
  }

  REQUIRE(struck);
  // Elastic-ish hit from a heavy paddle: v_out = (1 + e) * v_paddle.
  CHECK_THAT(-w.ball.vel.z, WithinRel(25.0 * 1.8, 1e-4));
  CHECK(w.ball.pos.z < paddle.pos.z);  // ball is in front of the paddle, not behind
}

TEST_CASE("the paddle is a disc, not an infinite plane", "[world][paddle]") {
  const Paddle paddle{.pos = {0, 1.0f, 1.5f}, .vel = {}, .normal = {0, 0, -1}};
  const real miss = paddle.radius + 0.05f;

  World w = world_with_ball({miss, 1.0f, 1.0f}, {0, 0, 8});
  w.ball_params.gravity = {};
  const ContactMask seen = run(w, std::span<const Paddle>{&paddle, 1}, 0.15f);
  CHECK_FALSE((seen & contact::kPaddle0));
  CHECK(w.ball.pos.z > 1.5f);
}

TEST_CASE("both faces of the paddle are playable", "[world][paddle]") {
  const Paddle paddle{.pos = {0, 1.0f, 1.5f}, .vel = {}, .normal = {0, 0, -1}};
  World w = world_with_ball({0, 1.0f, 2.0f}, {0, 0, -8});  // approaches the back face
  w.ball_params.gravity = {};
  const ContactMask seen = run(w, std::span<const Paddle>{&paddle, 1}, 0.15f);
  CHECK((seen & contact::kPaddle0));
  CHECK(w.ball.vel.z > 0);
}

TEST_CASE("a tilted paddle redirects the ball along its normal", "[world][paddle]") {
  // Open face (tilted up) sends a flat incoming ball upward.
  const Paddle paddle{.pos = {0, 1.0f, 1.5f}, .vel = {}, .normal = normalized(Vec3{0, 0.5f, -1})};
  World w = world_with_ball({0, 1.0f, 1.0f}, {0, 0, 8});
  w.ball_params.gravity = {};
  REQUIRE((run(w, std::span<const Paddle>{&paddle, 1}, 0.1f, kDt, contact::kPaddle0) &
           contact::kPaddle0));
  CHECK(w.ball.vel.y > 1.0f);
  CHECK(w.ball.vel.z < 0);
}

TEST_CASE("the net stops a low ball and lets a high one through", "[world][net]") {
  World low = world_with_ball({0, 0.85f, 0.5f}, {0, 0, -6});
  low.ball_params.gravity = {};
  CHECK((run(low, {}, 0.3f) & contact::kNet));
  CHECK(low.ball.pos.z > 0);  // stayed on its own side

  World high = world_with_ball({0, 1.0f, 0.5f}, {0, 0, -6});
  high.ball_params.gravity = {};
  CHECK_FALSE((run(high, {}, 0.3f) & contact::kNet));
  CHECK(high.ball.pos.z < 0);
}

TEST_CASE("clipping the top of the net deflects the ball upward", "[world][net]") {
  World w = world_with_ball({0, 0, 0.5f}, {0, 0, -6});
  w.ball_params.gravity = {};
  w.ball.pos.y = w.table.height + w.net.height + 0.012f;  // centre just above the tape
  REQUIRE((run(w, {}, 0.3f, kDt, contact::kNet) & contact::kNet));
  CHECK(w.ball.vel.y > 0);
  CHECK(w.ball.vel.z < 0);  // still carries over
}

TEST_CASE("the simulation is deterministic", "[world]") {
  auto simulate = [] {
    World w = world_with_ball({0.1f, 1.1f, -1.2f}, {0.4f, 2.0f, 4.2f}, {80, 15, -5});
    const std::array<Paddle, 1> paddle{
        Paddle{.pos = {0.2f, 0.95f, 1.5f}, .vel = {0, 0, -2}, .normal = normalized(Vec3{0, 0.3f, -1})}};
    for (int i = 0; i < 3000; ++i) step(w, paddle, kDt);
    return w.ball;
  };
  const BallState a = simulate(), b = simulate();
  CHECK(std::memcmp(&a, &b, sizeof(BallState)) == 0);
}
