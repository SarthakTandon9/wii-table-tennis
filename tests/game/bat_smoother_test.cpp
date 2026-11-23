#include "game/bat_smoother.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

using namespace pp;

namespace {
constexpr real kDt = 1e-3f;

struct Noise {
  std::uint32_t state = 0x51EDu;
  real next(real amplitude) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (static_cast<real>(state >> 8) / 8388608.0f - 1.0f) * amplitude;
  }
};
}  // namespace

TEST_CASE("a still bat with camera noise on it comes out nearly still", "[smoother]") {
  BatSmoother smoother;
  Noise noise;
  const Vec3 rest{0.2f, 0.95f, 1.5f};
  double in_sq = 0, out_sq = 0, shake_in = 0, shake_out = 0;
  Vec3 last_in = rest, last_out = rest;
  Vec3 held = rest;
  for (int i = 0; i < 4000; ++i) {
    // The noise changes 30 times a second, as camera frames do.
    if (i % 33 == 0) held = rest + Vec3{noise.next(0.008f), noise.next(0.008f), 0};
    const Vec3 out = smoother.update(held, kDt);
    if (i > 1000) {
      in_sq += static_cast<double>(length_sq(held - rest));
      out_sq += static_cast<double>(length_sq(out - rest));
      // Shake is what the eye sees: how far the bat jumps from one moment to the next.
      shake_in = std::max(shake_in, static_cast<double>(length(held - last_in)));
      shake_out = std::max(shake_out, static_cast<double>(length(out - last_out)));
    }
    last_in = held;
    last_out = out;
  }
  CHECK(out_sq < in_sq / 4.0);          // less than half the wander (RMS)
  CHECK(shake_out < shake_in / 20.0);   // and no sudden jumps at all
}

TEST_CASE("a bat that really moves is followed closely", "[smoother]") {
  BatSmoother smoother;
  Vec3 hand{-0.5f, 0.95f, 1.5f};
  for (int i = 0; i < 500; ++i) smoother.update(hand, kDt);
  // Across the table at 2.5 m/s.
  real worst_lag = 0;
  Vec3 out;
  for (int i = 0; i < 400; ++i) {
    hand.x += 2.5f * kDt;
    out = smoother.update(hand, kDt);
    if (i > 150) worst_lag = std::max(worst_lag, hand.x - out.x);
  }
  CHECK(worst_lag < 0.12f);  // a few centimetres behind, not half the table
  // And it arrives: no residue once the hand stops.
  for (int i = 0; i < 1500; ++i) out = smoother.update(hand, kDt);
  CHECK(length(out - hand) < 0.003f);
}
