#include "physics/ball.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>

using namespace pp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr Vec3 kUp{0, 1, 0};

BallParams vacuum() {
  BallParams p;
  p.air_density = 0;
  return p;
}

real mechanical_energy(const BallState& s, const BallParams& p) {
  return p.mass * (real{0.5} * length_sq(s.vel) - dot(p.gravity, s.pos));
}

real total_kinetic(const BallState& s, const BallParams& p) {
  const real inertia = real{2} / real{3} * p.mass * p.radius * p.radius;
  return real{0.5} * p.mass * length_sq(s.vel) + real{0.5} * inertia * length_sq(s.spin);
}

Vec3 contact_velocity(const BallState& s, const BallParams& p, Vec3 n) {
  const Vec3 tangential = s.vel - dot(s.vel, n) * n;
  return tangential + cross(s.spin, -p.radius * n);
}

void check_near(Vec3 a, Vec3 b, double tol) {
  CHECK_THAT(a.x, WithinAbs(static_cast<double>(b.x), tol));
  CHECK_THAT(a.y, WithinAbs(static_cast<double>(b.y), tol));
  CHECK_THAT(a.z, WithinAbs(static_cast<double>(b.z), tol));
}

}  // namespace

// ------------------------------------------------------------------- flight

TEST_CASE("a ball at rest accelerates at exactly g", "[flight]") {
  const BallParams p;
  const BallState s{.pos = {0, 1, 0}, .vel = {}, .spin = {50, -20, 300}};
  CHECK(acceleration(s, p) == p.gravity);
}

TEST_CASE("drag opposes velocity and is parallel to it", "[flight]") {
  const BallParams p;
  const BallState s{.pos = {}, .vel = {3, 4, -12}, .spin = {}};
  const Vec3 drag = acceleration(s, p) - p.gravity;

  CHECK(dot(drag, s.vel) < 0);
  check_near(cross(drag, s.vel), {}, 1e-3);

  // Magnitude: k_d |v|^2.
  const real area = std::numbers::pi_v<real> * p.radius * p.radius;
  const real k_d = p.air_density * area * p.drag_coeff / (2 * p.mass);
  CHECK_THAT(length(drag), WithinRel(static_cast<double>(k_d * 169), 1e-4));
}

TEST_CASE("Magnus force is perpendicular to velocity and has the right sign", "[flight]") {
  const BallParams p;
  const Vec3 v{0, 0, -10};  // travelling away from the player

  const Vec3 no_spin = acceleration({.pos = {}, .vel = v, .spin = {}}, p);

  SECTION("topspin dips") {
    const Vec3 a = acceleration({.pos = {}, .vel = v, .spin = {-200, 0, 0}}, p);
    CHECK(a.y < no_spin.y);
    CHECK_THAT(dot(a - no_spin, v), WithinAbs(0.0, 1e-3));
  }
  SECTION("backspin floats") {
    const Vec3 a = acceleration({.pos = {}, .vel = v, .spin = {200, 0, 0}}, p);
    CHECK(a.y > no_spin.y);
  }
  SECTION("sidespin about +y curves toward -x") {
    const Vec3 a = acceleration({.pos = {}, .vel = v, .spin = {0, 200, 0}}, p);
    CHECK(a.x < 0);
    CHECK_THAT(a.y, WithinAbs(static_cast<double>(no_spin.y), 1e-4));
  }
  SECTION("spin along the velocity does nothing") {
    const Vec3 a = acceleration({.pos = {}, .vel = v, .spin = {0, 0, 200}}, p);
    check_near(a, no_spin, 1e-4);
  }
}

TEST_CASE("in vacuum both integrators trace a parabola", "[flight][integrator]") {
  const BallParams p = vacuum();
  const BallState start{.pos = {0, 1, 0}, .vel = {1, 2, -3}, .spin = {0, 100, 0}};
  const real dt = 1e-3f;
  const int steps = 500;
  const real t = dt * static_cast<real>(steps);
  const Vec3 expected_pos = start.pos + start.vel * t + p.gravity * (real{0.5} * t * t);
  const Vec3 expected_vel = start.vel + p.gravity * t;

  SECTION("rk4") {
    BallState s = start;
    for (int i = 0; i < steps; ++i) step_rk4(s, p, dt);
    check_near(s.pos, expected_pos, 1e-4);
    check_near(s.vel, expected_vel, 1e-3);
    CHECK(s.spin == start.spin);
  }
  SECTION("semi-implicit euler (first order: error ~ g dt t / 2)") {
    BallState s = start;
    for (int i = 0; i < steps; ++i) step_semi_implicit(s, p, dt);
    check_near(s.pos, expected_pos, 5e-3);
    check_near(s.vel, expected_vel, 1e-3);
    CHECK(s.spin == start.spin);
  }
}

TEST_CASE("a dropped ball reaches terminal velocity", "[flight][integrator]") {
  const BallParams p;
  const real area = std::numbers::pi_v<real> * p.radius * p.radius;
  const real v_terminal =
      std::sqrt(2 * p.mass * length(p.gravity) / (p.air_density * area * p.drag_coeff));

  auto step = GENERATE(&step_rk4, &step_semi_implicit);
  BallState s{.pos = {0, 100, 0}, .vel = {}, .spin = {}};
  for (int i = 0; i < 8000; ++i) step(s, p, 1e-3f);

  CHECK(s.vel.y < 0);
  CHECK_THAT(-s.vel.y, WithinRel(static_cast<double>(v_terminal), 0.01));
  CHECK_THAT(s.vel.x, WithinAbs(0.0, 1e-6));
  CHECK_THAT(s.vel.z, WithinAbs(0.0, 1e-6));
}

TEST_CASE("mechanical energy never increases in flight", "[flight][integrator]") {
  // Drag removes energy; Magnus does no work (it is perpendicular to v).
  const BallParams p;
  BallState s{.pos = {0, 1, 1.3f}, .vel = {0.5f, 3, -9}, .spin = {-150, 40, 0}};
  real previous = mechanical_energy(s, p);
  for (int i = 0; i < 600; ++i) {
    step_rk4(s, p, 1e-3f);
    const real e = mechanical_energy(s, p);
    REQUIRE(e <= previous + 1e-6f);
    previous = e;
  }
  CHECK(previous < mechanical_energy({.pos = {0, 1, 1.3f}, .vel = {0.5f, 3, -9}, .spin = {}}, p));
}

TEST_CASE("rk4 beats semi-implicit euler at a coarse step", "[flight][integrator]") {
  const BallParams p;
  const BallState start{.pos = {0, 1, 1.3f}, .vel = {0.5f, 3, -9}, .spin = {-150, 40, 0}};

  BallState reference = start;
  for (int i = 0; i < 4000; ++i) step_rk4(reference, p, 1e-4f);

  BallState coarse_rk4 = start, coarse_euler = start;
  for (int i = 0; i < 40; ++i) {
    step_rk4(coarse_rk4, p, 1e-2f);
    step_semi_implicit(coarse_euler, p, 1e-2f);
  }

  const real err_rk4 = length(coarse_rk4.pos - reference.pos);
  const real err_euler = length(coarse_euler.pos - reference.pos);
  CHECK(err_rk4 < 1e-3f);
  CHECK(err_rk4 * 10 < err_euler);
}

// ------------------------------------------------------------------- bounce

TEST_CASE("a spinless vertical drop rebounds at e times the speed", "[bounce]") {
  const BallParams p;
  const SurfaceParams table;
  BallState s{.pos = {}, .vel = {0, -3, 0}, .spin = {}};
  bounce(s, p, kUp, table);
  check_near(s.vel, {0, 3 * table.restitution, 0}, 1e-5);
  check_near(s.spin, {}, 1e-4);
}

TEST_CASE("bounce ignores a ball that is moving away from the surface", "[bounce]") {
  const BallParams p;
  const BallState before{.pos = {}, .vel = {1, 2, -3}, .spin = {10, 20, 30}};
  BallState s = before;
  bounce(s, p, kUp, {});
  CHECK(s.vel == before.vel);
  CHECK(s.spin == before.spin);
}

TEST_CASE("sliding bounce: a fast flat spinless ball slows and picks up topspin", "[bounce]") {
  // mu (1+e) |v_n| = 0.25 * 1.9 * 3 = 1.425  <  (2/5) * 5 = 2  -> sliding
  const BallParams p;
  const SurfaceParams table;
  BallState s{.pos = {}, .vel = {0, -3, -5}, .spin = {}};
  bounce(s, p, kUp, table);

  CHECK_THAT(s.vel.y, WithinAbs(2.7, 1e-5));
  CHECK_THAT(s.vel.z, WithinAbs(-5.0 + 1.425, 1e-4));
  CHECK_THAT(s.vel.x, WithinAbs(0.0, 1e-6));
  CHECK(s.spin.x < 0);  // topspin for a ball heading toward -z
  CHECK(length(contact_velocity(s, p, kUp)) > 0.1f);  // still sliding
}

TEST_CASE("gripping bounce: a steep ball leaves with zero contact velocity", "[bounce]") {
  // mu (1+e) |v_n| = 0.25 * 1.9 * 5 = 2.375  >=  (2/5) * 1 = 0.4  -> grips
  const BallParams p;
  BallState s{.pos = {}, .vel = {0.6f, -5, -0.8f}, .spin = {}};
  bounce(s, p, kUp, {});
  check_near(contact_velocity(s, p, kUp), {}, 1e-4);
  // (2/5) of the tangential velocity is removed.
  CHECK_THAT(s.vel.x, WithinAbs(0.6 * 0.6, 1e-4));
  CHECK_THAT(s.vel.z, WithinAbs(-0.8 * 0.6, 1e-4));
}

TEST_CASE("topspin kicks forward off the table, backspin brakes", "[bounce]") {
  const BallParams p;
  const Vec3 v{0, -3, -5};

  BallState top{.pos = {}, .vel = v, .spin = {-300, 0, 0}};
  BallState back{.pos = {}, .vel = v, .spin = {300, 0, 0}};
  bounce(top, p, kUp, {});
  bounce(back, p, kUp, {});

  CHECK(-top.vel.z > 5.0f);   // faster than it came in
  CHECK(-back.vel.z < 5.0f);  // slower than it came in
  CHECK(std::abs(top.spin.x) < 300.0f);
  CHECK(std::abs(back.spin.x) < 300.0f);
}

TEST_CASE("bounce conserves angular momentum about the contact point", "[bounce]") {
  // Every impulse acts at the contact point, so  I dw = (-r n) x (m dv),
  // which for a hollow shell reduces to  (2/3) r dw = -(n x dv).
  const BallParams p;
  const Vec3 n = normalized(GENERATE(Vec3{0, 1, 0}, Vec3{0.2f, 1, -0.4f}, Vec3{0, 0.3f, 1}));
  const BallState before = GENERATE(
      BallState{.pos = {}, .vel = {0, -3, -5}, .spin = {}},
      BallState{.pos = {}, .vel = {2, -4, -7}, .spin = {-250, 60, 30}},
      BallState{.pos = {}, .vel = {-1, -6, 0.5f}, .spin = {300, -100, 0}},
      BallState{.pos = {}, .vel = {0.3f, -1, -12}, .spin = {0, 400, 0}});
  if (dot(before.vel, n) >= 0) return;

  BallState s = before;
  bounce(s, p, n, {});

  const Vec3 lhs = (real{2} / real{3}) * p.radius * (s.spin - before.spin);
  const Vec3 rhs = -cross(n, s.vel - before.vel);
  check_near(lhs, rhs, 2e-3);

  // Normal component always reflects with restitution e.
  CHECK_THAT(dot(s.vel, n), WithinAbs(static_cast<double>(-0.9f * dot(before.vel, n)), 1e-3));
  // A passive surface cannot add energy.
  CHECK(total_kinetic(s, p) <= total_kinetic(before, p) * 1.0001f);
}
