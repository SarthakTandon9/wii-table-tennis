#include "physics/ball.hpp"

#include <numbers>

namespace pp {

namespace {

// Acceleration depends on velocity only (position does not appear, spin is
// constant in flight), which keeps the RK4 stages cheap.
struct FlightCoeffs {
  real k_drag;
  real k_magnus;
};

FlightCoeffs coeffs(const BallParams& p) {
  const real area = std::numbers::pi_v<real> * p.radius * p.radius;
  const real half_rho_a_over_m = p.air_density * area / (real{2} * p.mass);
  return {half_rho_a_over_m * p.drag_coeff, half_rho_a_over_m * p.radius * p.magnus_coeff};
}

Vec3 accel(Vec3 vel, Vec3 spin, Vec3 gravity, FlightCoeffs k) {
  return gravity - (k.k_drag * length(vel)) * vel + k.k_magnus * cross(spin, vel);
}

}  // namespace

Vec3 acceleration(const BallState& s, const BallParams& p) {
  return accel(s.vel, s.spin, p.gravity, coeffs(p));
}

void step_semi_implicit(BallState& s, const BallParams& p, real dt) {
  s.vel += accel(s.vel, s.spin, p.gravity, coeffs(p)) * dt;
  s.pos += s.vel * dt;  // uses the NEW velocity: that is the "semi-implicit" part
}

void step_rk4(BallState& s, const BallParams& p, real dt) {
  const FlightCoeffs k = coeffs(p);
  const real half = dt * 0.5f;

  // State is (pos, vel); derivative is (vel, a(vel)).
  const Vec3 v1 = s.vel;
  const Vec3 a1 = accel(v1, s.spin, p.gravity, k);
  const Vec3 v2 = s.vel + a1 * half;
  const Vec3 a2 = accel(v2, s.spin, p.gravity, k);
  const Vec3 v3 = s.vel + a2 * half;
  const Vec3 a3 = accel(v3, s.spin, p.gravity, k);
  const Vec3 v4 = s.vel + a3 * dt;
  const Vec3 a4 = accel(v4, s.spin, p.gravity, k);

  const real sixth = dt / real{6};
  s.pos += (v1 + real{2} * (v2 + v3) + v4) * sixth;
  s.vel += (a1 + real{2} * (a2 + a3) + a4) * sixth;
}

void bounce(BallState& s, const BallParams& p, Vec3 n, const SurfaceParams& surface) {
  const real v_n = dot(s.vel, n);
  if (v_n >= real{0}) return;  // separating or sliding along: nothing to resolve

  // Impulses below are per unit mass (i.e. velocity changes).
  const real j_n = -(real{1} + surface.restitution) * v_n;  // > 0

  const Vec3 contact_vel = (s.vel - v_n * n) + cross(s.spin, -p.radius * n);
  const real contact_speed = length(contact_vel);

  Vec3 j_t{};
  if (contact_speed > real{0}) {
    constexpr real kGrip = real{2} / real{5};  // hollow shell, see header
    const real max_friction = surface.friction * j_n;
    if (max_friction < kGrip * contact_speed) {
      j_t = -(max_friction / contact_speed) * contact_vel;  // sliding
    } else {
      j_t = -kGrip * contact_vel;  // friction is enough to stop the contact point
    }
  }

  const Vec3 dv = j_n * n + j_t;
  s.vel += dv;
  // I dw = (-r n) x (m dv), I = (2/3) m r^2  =>  dw = (3 / 2r) * (dv x n)
  s.spin += (1.5f / p.radius) * cross(dv, n);
}

}  // namespace pp
