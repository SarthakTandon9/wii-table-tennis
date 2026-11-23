#pragma once
// A one-dimensional Kalman filter with a constant-velocity model.
//
// The problem it solves: the camera gives a hand position 30 times a second,
// each one noisy (the hand "shivers" by a few pixels even when held still)
// and each one already old by the time it arrives. Using the samples directly
// gives a paddle that shakes and trails behind the hand.
//
// The filter keeps a belief about two things, position p and velocity v,
// together with how unsure it is about them (the covariance P). Two steps:
//
//   predict(dt)   time passes: p moves by v*dt, and the uncertainty GROWS,
//                 because the hand may have accelerated meanwhile.
//   update(z)     a measurement arrives: move the belief toward it, by an
//                 amount (the gain K) that depends on which is more
//                 trustworthy right now, the belief or the measurement.
//
// Velocity is never measured. It is inferred, because a position that keeps
// landing ahead of the prediction can only mean the hand is moving. Having a
// velocity is what makes it possible to extrapolate past the last sample and
// cancel the camera's delay.
//
// Jitter against lag: `accel_sigma` says how hard the hand might be
// accelerating. Small: the filter trusts its own prediction, output is smooth
// but slow to follow a sudden move. Large: it follows at once but lets the
// noise through. No fixed value is good for both a still hand and a swing, so
// the caller scales it with speed (the idea behind the "1 Euro" filter used
// widely for hand tracking): calm when the hand is calm, quick when it moves.

namespace pp {

struct KalmanCV {
  float p = 0, v = 0;                   // position, velocity
  float P00 = 1, P01 = 0, P11 = 1;      // covariance [[P00, P01], [P01, P11]]

  void reset(float position, float position_var, float velocity_var) {
    p = position;
    v = 0;
    P00 = position_var;
    P01 = 0;
    P11 = velocity_var;
  }

  // x <- F x,  P <- F P F' + Q,   F = [[1, dt], [0, 1]]
  // Q is the effect of an unknown constant acceleration over dt.
  void predict(float dt, float accel_sigma) {
    p += v * dt;
    const float q = accel_sigma * accel_sigma;
    const float dt2 = dt * dt;
    P00 += dt * (2.0f * P01 + dt * P11) + q * dt2 * dt2 * 0.25f;
    P01 += dt * P11 + q * dt2 * dt * 0.5f;
    P11 += q * dt2;
  }

  // Measurement z of the position, with variance r.
  void update(float z, float r) {
    const float innovation = z - p;      // how wrong the prediction was
    const float S = P00 + r;             // how wrong we expected it to be
    const float K0 = P00 / S, K1 = P01 / S;
    p += K0 * innovation;
    v += K1 * innovation;
    const float P00_old = P00, P01_old = P01;
    P00 = (1.0f - K0) * P00_old;
    P01 = (1.0f - K0) * P01_old;
    P11 -= K1 * P01_old;
  }

  // Where it will be `ahead` seconds after the last update.
  float extrapolate(float ahead) const { return p + v * ahead; }
};

}  // namespace pp
