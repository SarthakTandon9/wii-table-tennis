#pragma once
// Ball flight and impact.
//
// World frame: right-handed, +y up, metres / seconds / radians. The player
// stands at +z and hits toward -z.
//
// FLIGHT
//   a = g  -  k_d * |v| * v  +  k_m * (w x v)
//   k_d = rho * A * C_d / (2 m)          quadratic drag
//   k_m = rho * A * r * C_m / (2 m)      Magnus (spin) force
//   A   = pi * r^2
// Spin w is held constant during flight (no spin decay yet).
// Sign check: a ball travelling toward -z with topspin has w along -x, and
// w x v then points down. Topspin dips.
//
// BOUNCE (instantaneous impulse against a rigid plane with unit normal n)
// The ball is a thin hollow shell: I = (2/3) m r^2.
//   v_n  = dot(v, n)                      (negative when approaching)
//   v_c  = (v - v_n n) + w x (-r n)       velocity of the contact point
//   Normal:      v_n' = -e * v_n
//   Tangential:  Coulomb friction, impulse opposes v_c.
//     Sliding  if mu (1+e) |v_n| <  (2/5) |v_c| :  J_t = -mu (1+e) m |v_n| * v_c/|v_c|
//     Gripping otherwise:                          J_t = -(2/5) m v_c   (contact point stops)
//   Then  m dv = J,   I dw = (-r n) x J.
// Where 2/5 comes from: a tangential impulse J changes the contact velocity by
// J/m + r^2 J / I = (J/m)(1 + 3/2). Setting that equal to -v_c gives
// J = -(2/5) m v_c. (A solid sphere, I = 2/5 m r^2, would give 2/7.)

#include "core/types.hpp"
#include "math/vec3.hpp"

namespace pp {

struct BallState {
  Vec3 pos;   // m
  Vec3 vel;   // m/s
  Vec3 spin;  // rad/s, angular velocity vector
};

struct BallParams {
  real mass = 0.0027f;        // kg   (ITTF: 2.7 g)
  real radius = 0.020f;       // m    (ITTF: 40 mm diameter)
  real air_density = 1.2f;    // kg/m^3
  real drag_coeff = 0.40f;    // C_d
  real magnus_coeff = 0.60f;  // C_m
  Vec3 gravity{0.0f, -9.81f, 0.0f};
};

struct SurfaceParams {
  real restitution = 0.90f;  // e
  real friction = 0.25f;     // mu
};

// Acceleration of the ball in free flight.
Vec3 acceleration(const BallState& s, const BallParams& p);

// Semi-implicit (symplectic) Euler: v += a dt, then x += v dt. First order.
void step_semi_implicit(BallState& s, const BallParams& p, real dt);
// Classic RK4 on (pos, vel). Fourth order, four acceleration evaluations.
void step_rk4(BallState& s, const BallParams& p, real dt);

// Resolve an impact against a static plane with unit normal `n`.
// Modifies vel and spin only. Does nothing if the ball is moving away.
void bounce(BallState& s, const BallParams& p, Vec3 n, const SurfaceParams& surface);

}  // namespace pp
