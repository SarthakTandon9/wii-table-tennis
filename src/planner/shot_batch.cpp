#include "planner/shot_batch.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace pp {

namespace {

// Everything constant across the batch, computed once.
struct Consts {
  float gx, gy, gz;
  float k_drag, k_magnus;
  float h, half_h, sixth_h;
  float land_y, net_line;  // net_line = net_top + ball radius: centre height that just clears
};

Consts make_consts(const ShotQuery& q) {
  const float area = std::numbers::pi_v<float> * q.ball.radius * q.ball.radius;
  const float k = q.ball.air_density * area / (2.0f * q.ball.mass);
  return {q.ball.gravity.x, q.ball.gravity.y, q.ball.gravity.z,
          k * q.ball.drag_coeff, k * q.ball.radius * q.ball.magnus_coeff,
          q.dt, q.dt * 0.5f, q.dt / 6.0f,
          q.land_y, q.net_top + q.ball.radius};
}

// ---- the scalar kernel: one RK4 step for one ball ----
// Written on plain floats (not Vec3) and force-inlined, so that when it is
// called inside a loop over array elements the compiler sees straight-line
// arithmetic it can turn into vector instructions.

[[gnu::always_inline]] inline void accel(float vx, float vy, float vz, float wx, float wy,
                                         float wz, const Consts& c, float& ax, float& ay,
                                         float& az) {
  const float drag = c.k_drag * std::sqrt(vx * vx + vy * vy + vz * vz);
  ax = c.gx - drag * vx + c.k_magnus * (wy * vz - wz * vy);
  ay = c.gy - drag * vy + c.k_magnus * (wz * vx - wx * vz);
  az = c.gz - drag * vz + c.k_magnus * (wx * vy - wy * vx);
}

[[gnu::always_inline]] inline void rk4(float& px, float& py, float& pz, float& vx, float& vy,
                                       float& vz, float wx, float wy, float wz, const Consts& c) {
  float a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z, a4x, a4y, a4z;
  accel(vx, vy, vz, wx, wy, wz, c, a1x, a1y, a1z);
  const float v2x = vx + a1x * c.half_h, v2y = vy + a1y * c.half_h, v2z = vz + a1z * c.half_h;
  accel(v2x, v2y, v2z, wx, wy, wz, c, a2x, a2y, a2z);
  const float v3x = vx + a2x * c.half_h, v3y = vy + a2y * c.half_h, v3z = vz + a2z * c.half_h;
  accel(v3x, v3y, v3z, wx, wy, wz, c, a3x, a3y, a3z);
  const float v4x = vx + a3x * c.h, v4y = vy + a3y * c.h, v4z = vz + a3z * c.h;
  accel(v4x, v4y, v4z, wx, wy, wz, c, a4x, a4y, a4z);

  px += (vx + 2.0f * (v2x + v3x) + v4x) * c.sixth_h;
  py += (vy + 2.0f * (v2y + v3y) + v4y) * c.sixth_h;
  pz += (vz + 2.0f * (v2z + v3z) + v4z) * c.sixth_h;
  vx += (a1x + 2.0f * (a2x + a3x) + a4x) * c.sixth_h;
  vy += (a1y + 2.0f * (a2y + a3y) + a4y) * c.sixth_h;
  vz += (a1z + 2.0f * (a2z + a3z) + a4z) * c.sixth_h;
}

// ---- shared SoA sweep ----
// Advances candidates [begin, end) through steps [step_begin, step_end).
// State lives in the caller's arrays. Returns how many have landed.
//
// No `if` in the loop body: a branch per element would stop the compiler
// vectorising. Every decision is a comparison producing 0 or 1, and results
// are chosen with ?: (a "select" in vector code). A candidate that has
// already landed is still integrated; its outputs simply stop changing.
struct SweepState {
  float *px, *py, *pz, *vx, *vy, *vz;
  std::uint32_t *landed, *crossed;
};

std::size_t sweep(const Consts& c, ShotBatch& b, const SweepState& s, std::size_t begin,
                  std::size_t end, int step_begin, int step_end, std::size_t state_offset) {
  const float* __restrict wx = b.wx.data();
  const float* __restrict wy = b.wy.data();
  const float* __restrict wz = b.wz.data();
  float* __restrict out_x = b.land_x.data();
  float* __restrict out_z = b.land_z.data();
  float* __restrict out_t = b.time.data();
  float* __restrict out_c = b.clearance.data();
  float* __restrict px = s.px;
  float* __restrict py = s.py;
  float* __restrict pz = s.pz;
  float* __restrict vx = s.vx;
  float* __restrict vy = s.vy;
  float* __restrict vz = s.vz;
  std::uint32_t* __restrict landed = s.landed;
  std::uint32_t* __restrict crossed = s.crossed;

  for (int step = step_begin; step < step_end; ++step) {
    const float t0 = static_cast<float>(step) * c.h;
    // Left to itself, clang does NOT vectorise this loop: its cost model
    // reports "vectorization is not beneficial" (check with
    // -Rpass-missed=loop-vectorize). The model is wary of the two divisions
    // and the many selects. The pragma overrules it. Never assume a loop was
    // vectorised because it looks vectorisable; ask the compiler.
#pragma clang loop vectorize(enable) interleave(disable)
    for (std::size_t i = begin; i < end; ++i) {
      const std::size_t k = i - state_offset;  // state arrays may be block-local
      float x = px[k], y = py[k], z = pz[k], ux = vx[k], uy = vy[k], uz = vz[k];
      const float x0 = x, y0 = y, z0 = z;
      rk4(x, y, z, ux, uy, uz, wx[i], wy[i], wz[i], c);

      const std::uint32_t live = landed[k] == 0;

      // Crossing the net plane z = 0: note the clearance, once.
      const std::uint32_t cross = ((z0 > 0) != (z > 0)) & live & (crossed[k] == 0);
      const float at_net = y0 + (y - y0) * (z0 / (z0 - z));
      out_c[i] = cross ? at_net - c.net_line : out_c[i];
      crossed[k] |= cross;

      // Coming down through table height: note where and when, once.
      const std::uint32_t land = (uy < 0) & (y0 > c.land_y) & (y <= c.land_y) & live;
      const float f = (y0 - c.land_y) / (y0 - y);
      out_x[i] = land ? x0 + (x - x0) * f : out_x[i];
      out_z[i] = land ? z0 + (z - z0) * f : out_z[i];
      out_t[i] = land ? t0 + f * c.h : out_t[i];
      landed[k] |= land;

      px[k] = x; py[k] = y; pz[k] = z;
      vx[k] = ux; vy[k] = uy; vz[k] = uz;
    }
  }

  std::size_t done = 0;
  for (std::size_t i = begin; i < end; ++i) done += landed[i - state_offset];
  return done;
}

void reset_outputs(ShotBatch& b, std::size_t begin, std::size_t end) {
  std::fill(b.land_x.begin() + static_cast<std::ptrdiff_t>(begin),
            b.land_x.begin() + static_cast<std::ptrdiff_t>(end), 0.0f);
  std::fill(b.land_z.begin() + static_cast<std::ptrdiff_t>(begin),
            b.land_z.begin() + static_cast<std::ptrdiff_t>(end), 0.0f);
  std::fill(b.time.begin() + static_cast<std::ptrdiff_t>(begin),
            b.time.begin() + static_cast<std::ptrdiff_t>(end), -1.0f);
  std::fill(b.clearance.begin() + static_cast<std::ptrdiff_t>(begin),
            b.clearance.begin() + static_cast<std::ptrdiff_t>(end), -100.0f);
}

constexpr int kCheckEvery = 16;  // steps between "has everything landed?" checks

}  // namespace

void ShotBatch::resize(std::size_t n) {
  count = n;
  const std::size_t padded_size = (n + 3) & ~std::size_t{3};
  for (auto* v : {&vx, &vy, &vz, &wx, &wy, &wz, &land_x, &land_z, &time, &clearance}) {
    v->assign(padded_size, 0.0f);
  }
}

void ShotBatch::set(std::size_t i, const ShotCandidate& c) {
  vx[i] = c.vel.x; vy[i] = c.vel.y; vz[i] = c.vel.z;
  wx[i] = c.spin.x; wy[i] = c.spin.y; wz[i] = c.spin.z;
}

// ----------------------------------------------------------------------- AoS

void simulate_aos(const ShotQuery& q, std::span<const ShotCandidate> in,
                  std::span<ShotOutcome> out) {
  const Consts c = make_consts(q);
  for (std::size_t i = 0; i < in.size(); ++i) {
    float x = q.from.x, y = q.from.y, z = q.from.z;
    float ux = in[i].vel.x, uy = in[i].vel.y, uz = in[i].vel.z;
    ShotOutcome result;
    bool crossed = false;
    for (int step = 0; step < q.max_steps; ++step) {
      const float x0 = x, y0 = y, z0 = z;
      rk4(x, y, z, ux, uy, uz, in[i].spin.x, in[i].spin.y, in[i].spin.z, c);
      if (!crossed && (z0 > 0) != (z > 0)) {
        result.clearance = y0 + (y - y0) * (z0 / (z0 - z)) - c.net_line;
        crossed = true;
      }
      if (uy < 0 && y0 > c.land_y && y <= c.land_y) {
        const float f = (y0 - c.land_y) / (y0 - y);
        result.land_x = x0 + (x - x0) * f;
        result.land_z = z0 + (z - z0) * f;
        result.time = static_cast<float>(step) * c.h + f * c.h;
        break;  // the one advantage of doing them one at a time: stop early
      }
    }
    out[i] = result;
  }
}

// ------------------------------------------------------------------ SoA wide

void simulate_soa_wide(const ShotQuery& q, ShotBatch& b, std::size_t begin, std::size_t end) {
  const Consts c = make_consts(q);
  const std::size_t n = end - begin;
  reset_outputs(b, begin, end);

  // Six state arrays the size of the batch. For a large batch this is far
  // more than the L1 cache holds, and every time step reads and writes all
  // of it: the arithmetic is vectorised but the memory traffic is heavy.
  std::vector<float> px(n, q.from.x), py(n, q.from.y), pz(n, q.from.z);
  std::vector<float> vx(b.vx.begin() + static_cast<std::ptrdiff_t>(begin),
                        b.vx.begin() + static_cast<std::ptrdiff_t>(end));
  std::vector<float> vy(b.vy.begin() + static_cast<std::ptrdiff_t>(begin),
                        b.vy.begin() + static_cast<std::ptrdiff_t>(end));
  std::vector<float> vz(b.vz.begin() + static_cast<std::ptrdiff_t>(begin),
                        b.vz.begin() + static_cast<std::ptrdiff_t>(end));
  std::vector<std::uint32_t> landed(n, 0), crossed(n, 0);
  const SweepState s{px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data(),
                     landed.data(), crossed.data()};

  for (int step = 0; step < q.max_steps; step += kCheckEvery) {
    const int stop = std::min(step + kCheckEvery, q.max_steps);
    if (sweep(c, b, s, begin, end, step, stop, begin) == n) break;
  }
}

// --------------------------------------------------------------- SoA blocked

void simulate_soa_blocked(const ShotQuery& q, ShotBatch& b, std::size_t begin, std::size_t end) {
  const Consts c = make_consts(q);
  reset_outputs(b, begin, end);

  // Same sweep, but a block of 64 candidates is carried through ALL its time
  // steps before the next block is touched. A block's state is 64 * 8 values
  // = 2 KB: it stays in the L1 cache (and on the stack) for its whole flight.
  constexpr std::size_t kBlock = 64;
  float px[kBlock], py[kBlock], pz[kBlock], vx[kBlock], vy[kBlock], vz[kBlock];
  std::uint32_t landed[kBlock], crossed[kBlock];
  const SweepState s{px, py, pz, vx, vy, vz, landed, crossed};

  for (std::size_t block = begin; block < end; block += kBlock) {
    const std::size_t block_end = std::min(block + kBlock, end);
    const std::size_t n = block_end - block;
    for (std::size_t k = 0; k < n; ++k) {
      px[k] = q.from.x; py[k] = q.from.y; pz[k] = q.from.z;
      vx[k] = b.vx[block + k]; vy[k] = b.vy[block + k]; vz[k] = b.vz[block + k];
      landed[k] = 0; crossed[k] = 0;
    }
    for (int step = 0; step < q.max_steps; step += kCheckEvery) {
      const int stop = std::min(step + kCheckEvery, q.max_steps);
      if (sweep(c, b, s, block, block_end, step, stop, block) == n) break;
    }
  }
}

// ---------------------------------------------------------------------- NEON

#if defined(__aarch64__)

namespace {

// Four balls at once. A float32x4_t is one 128-bit register holding four
// floats; every operation below acts on all four lanes.
struct Vec4 {
  float32x4_t x, y, z;
};

[[gnu::always_inline]] inline Vec4 accel4(Vec4 v, Vec4 w, const Consts& c) {
  const float32x4_t speed_sq =
      vaddq_f32(vaddq_f32(vmulq_f32(v.x, v.x), vmulq_f32(v.y, v.y)), vmulq_f32(v.z, v.z));
  const float32x4_t drag = vmulq_n_f32(vsqrtq_f32(speed_sq), c.k_drag);
  const float32x4_t cx = vsubq_f32(vmulq_f32(w.y, v.z), vmulq_f32(w.z, v.y));
  const float32x4_t cy = vsubq_f32(vmulq_f32(w.z, v.x), vmulq_f32(w.x, v.z));
  const float32x4_t cz = vsubq_f32(vmulq_f32(w.x, v.y), vmulq_f32(w.y, v.x));
  return {vaddq_f32(vsubq_f32(vdupq_n_f32(c.gx), vmulq_f32(drag, v.x)), vmulq_n_f32(cx, c.k_magnus)),
          vaddq_f32(vsubq_f32(vdupq_n_f32(c.gy), vmulq_f32(drag, v.y)), vmulq_n_f32(cy, c.k_magnus)),
          vaddq_f32(vsubq_f32(vdupq_n_f32(c.gz), vmulq_f32(drag, v.z)), vmulq_n_f32(cz, c.k_magnus))};
}

[[gnu::always_inline]] inline Vec4 axpy(Vec4 a, Vec4 b, float s) {  // a + b * s
  return {vaddq_f32(a.x, vmulq_n_f32(b.x, s)), vaddq_f32(a.y, vmulq_n_f32(b.y, s)),
          vaddq_f32(a.z, vmulq_n_f32(b.z, s))};
}

// k1 + 2 (k2 + k3) + k4
[[gnu::always_inline]] inline float32x4_t rk_sum(float32x4_t k1, float32x4_t k2, float32x4_t k3,
                                                 float32x4_t k4) {
  return vaddq_f32(vaddq_f32(k1, vmulq_n_f32(vaddq_f32(k2, k3), 2.0f)), k4);
}

}  // namespace

// Flights differ in length. If four balls were simply run until all four had
// landed, three lanes would idle while the slowest finished, and a quarter of
// the vector's work would be wasted (the first version of this function did
// exactly that, and was only 2.5x faster than scalar code).
//
// Instead each lane is refilled the moment its ball lands: its result is
// written out and the next waiting candidate is loaded into that lane. All
// four lanes stay busy until the batch runs out. The refill itself is scalar
// and clumsy (spill the registers to memory, patch one lane, reload), but it
// happens once per ball, against a couple of hundred vector steps per ball.
void simulate_neon(const ShotQuery& q, ShotBatch& b, std::size_t begin, std::size_t end) {
  if (begin >= end) return;
  const Consts c = make_consts(q);
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t land_y = vdupq_n_f32(c.land_y);
  const float32x4_t max_steps = vdupq_n_f32(static_cast<float>(q.max_steps));
  constexpr std::size_t kNone = ~std::size_t{0};

  // Per-lane state, in memory form. Used only when refilling.
  alignas(16) float m_px[4], m_py[4], m_pz[4], m_vx[4], m_vy[4], m_vz[4];
  alignas(16) float m_wx[4], m_wy[4], m_wz[4], m_step[4];
  alignas(16) float m_ox[4], m_oz[4], m_ot[4], m_oc[4];
  alignas(16) std::uint32_t m_crossed[4], m_active[4], m_finished[4];
  std::size_t lane_index[4];
  std::size_t next = begin;

  const auto load_lane = [&](int lane) {
    if (next >= end) {  // nothing left: park the lane on a ball that just sits there
      lane_index[lane] = kNone;
      m_active[lane] = 0;
      m_vx[lane] = m_vy[lane] = m_vz[lane] = m_wx[lane] = m_wy[lane] = m_wz[lane] = 0;
    } else {
      lane_index[lane] = next;
      m_active[lane] = 0xFFFFFFFFu;
      m_vx[lane] = b.vx[next]; m_vy[lane] = b.vy[next]; m_vz[lane] = b.vz[next];
      m_wx[lane] = b.wx[next]; m_wy[lane] = b.wy[next]; m_wz[lane] = b.wz[next];
      ++next;
    }
    m_px[lane] = q.from.x; m_py[lane] = q.from.y; m_pz[lane] = q.from.z;
    m_step[lane] = 0;
    m_ox[lane] = 0; m_oz[lane] = 0; m_ot[lane] = -1.0f; m_oc[lane] = -100.0f;
    m_crossed[lane] = 0;
  };
  for (int lane = 0; lane < 4; ++lane) load_lane(lane);

  Vec4 p{vld1q_f32(m_px), vld1q_f32(m_py), vld1q_f32(m_pz)};
  Vec4 v{vld1q_f32(m_vx), vld1q_f32(m_vy), vld1q_f32(m_vz)};
  Vec4 w{vld1q_f32(m_wx), vld1q_f32(m_wy), vld1q_f32(m_wz)};
  float32x4_t step = vld1q_f32(m_step);
  float32x4_t out_x = vld1q_f32(m_ox), out_z = vld1q_f32(m_oz);
  float32x4_t out_t = vld1q_f32(m_ot), out_c = vld1q_f32(m_oc);
  uint32x4_t crossed = vld1q_u32(m_crossed), active = vld1q_u32(m_active);

  // The time loop. Between refills, nothing is loaded from or stored to
  // memory: the state of four balls lives entirely in registers.
  while (vmaxvq_u32(active) != 0) {
    const Vec4 p0 = p;

    const Vec4 a1 = accel4(v, w, c);
    const Vec4 v2 = axpy(v, a1, c.half_h);
    const Vec4 a2 = accel4(v2, w, c);
    const Vec4 v3 = axpy(v, a2, c.half_h);
    const Vec4 a3 = accel4(v3, w, c);
    const Vec4 v4 = axpy(v, a3, c.h);
    const Vec4 a4 = accel4(v4, w, c);

    p = {vaddq_f32(p.x, vmulq_n_f32(rk_sum(v.x, v2.x, v3.x, v4.x), c.sixth_h)),
         vaddq_f32(p.y, vmulq_n_f32(rk_sum(v.y, v2.y, v3.y, v4.y), c.sixth_h)),
         vaddq_f32(p.z, vmulq_n_f32(rk_sum(v.z, v2.z, v3.z, v4.z), c.sixth_h))};
    v = {vaddq_f32(v.x, vmulq_n_f32(rk_sum(a1.x, a2.x, a3.x, a4.x), c.sixth_h)),
         vaddq_f32(v.y, vmulq_n_f32(rk_sum(a1.y, a2.y, a3.y, a4.y), c.sixth_h)),
         vaddq_f32(v.z, vmulq_n_f32(rk_sum(a1.z, a2.z, a3.z, a4.z), c.sixth_h))};

    // Net: the sign of z changed. Comparisons give all-ones or all-zeros per
    // lane. Crossing happens once per flight, so the division it needs is
    // behind a (well-predicted) branch instead of being paid on every step.
    const uint32x4_t side_changed = veorq_u32(vcgtq_f32(p0.z, zero), vcgtq_f32(p.z, zero));
    const uint32x4_t cross = vandq_u32(vandq_u32(side_changed, active), vmvnq_u32(crossed));
    if (vmaxvq_u32(cross) != 0) {
      const float32x4_t u = vdivq_f32(p0.z, vsubq_f32(p0.z, p.z));
      const float32x4_t at_net = vaddq_f32(p0.y, vmulq_f32(vsubq_f32(p.y, p0.y), u));
      // vbslq picks between two values lane by lane using the mask.
      out_c = vbslq_f32(cross, vsubq_f32(at_net, vdupq_n_f32(c.net_line)), out_c);
      crossed = vorrq_u32(crossed, cross);
    }

    // Landing: moving down, was above table height, now at or below it.
    const uint32x4_t land =
        vandq_u32(vandq_u32(vcltq_f32(v.y, zero), vcgtq_f32(p0.y, land_y)),
                  vandq_u32(vcleq_f32(p.y, land_y), active));
    const uint32x4_t timed_out = vandq_u32(vcgeq_f32(vaddq_f32(step, one), max_steps), active);
    const uint32x4_t finished = vorrq_u32(land, timed_out);

    if (vmaxvq_u32(finished) == 0) {  // the common case: carry on
      step = vaddq_f32(step, one);
      continue;
    }

    if (vmaxvq_u32(land) != 0) {
      const float32x4_t f = vdivq_f32(vsubq_f32(p0.y, land_y), vsubq_f32(p0.y, p.y));
      out_x = vbslq_f32(land, vaddq_f32(p0.x, vmulq_f32(vsubq_f32(p.x, p0.x), f)), out_x);
      out_z = vbslq_f32(land, vaddq_f32(p0.z, vmulq_f32(vsubq_f32(p.z, p0.z), f)), out_z);
      out_t = vbslq_f32(land, vmulq_n_f32(vaddq_f32(step, f), c.h), out_t);
    }
    step = vaddq_f32(step, one);

    // Refill: spill, write out the finished lanes, load new candidates, reload.
    vst1q_f32(m_px, p.x); vst1q_f32(m_py, p.y); vst1q_f32(m_pz, p.z);
    vst1q_f32(m_vx, v.x); vst1q_f32(m_vy, v.y); vst1q_f32(m_vz, v.z);
    vst1q_f32(m_wx, w.x); vst1q_f32(m_wy, w.y); vst1q_f32(m_wz, w.z);
    vst1q_f32(m_step, step);
    vst1q_f32(m_ox, out_x); vst1q_f32(m_oz, out_z);
    vst1q_f32(m_ot, out_t); vst1q_f32(m_oc, out_c);
    vst1q_u32(m_crossed, crossed);
    vst1q_u32(m_finished, finished);
    for (int lane = 0; lane < 4; ++lane) {
      if (m_finished[lane] == 0) continue;
      const std::size_t i = lane_index[lane];
      b.land_x[i] = m_ox[lane];
      b.land_z[i] = m_oz[lane];
      b.time[i] = m_ot[lane];
      b.clearance[i] = m_oc[lane];
      load_lane(lane);
    }
    p = {vld1q_f32(m_px), vld1q_f32(m_py), vld1q_f32(m_pz)};
    v = {vld1q_f32(m_vx), vld1q_f32(m_vy), vld1q_f32(m_vz)};
    w = {vld1q_f32(m_wx), vld1q_f32(m_wy), vld1q_f32(m_wz)};
    step = vld1q_f32(m_step);
    out_x = vld1q_f32(m_ox); out_z = vld1q_f32(m_oz);
    out_t = vld1q_f32(m_ot); out_c = vld1q_f32(m_oc);
    crossed = vld1q_u32(m_crossed);
    active = vld1q_u32(m_active);
  }
}

#else

void simulate_neon(const ShotQuery& q, ShotBatch& b, std::size_t begin, std::size_t end) {
  simulate_soa_blocked(q, b, begin, end);  // no NEON on this CPU
}

#endif

}  // namespace pp
