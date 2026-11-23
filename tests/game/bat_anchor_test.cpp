#include "game/bat_anchor.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pp;

namespace {
constexpr real kDt = 1e-3f;
}

TEST_CASE("slow hand movement places the bat", "[anchor]") {
  BatAnchor anchor;
  Vec3 out;
  for (int i = 0; i < 500; ++i) {
    const Vec3 hand{0.001f * static_cast<real>(i), 1.0f, 1.5f};
    out = anchor.update(hand, 0.3f, false, kDt);
    CHECK(length(out - hand) < 1e-4f);
  }
  CHECK_FALSE(anchor.holding());
}

TEST_CASE("a swing leaves the bat where it was before the hand took off", "[anchor]") {
  BatAnchor anchor;
  const Vec3 rest{0.10f, 0.95f, 1.5f};
  for (int i = 0; i < 400; ++i) anchor.update(rest, 0.1f, false, kDt);

  // The hand takes off to the right at 6 m/s. For the first 80 ms the speed
  // estimate has not caught up (it needs camera frames), so the bat is
  // dragged along; then the speed reads as fast and a swing goes live.
  Vec3 hand = rest, out;
  for (int i = 0; i < 80; ++i) {
    hand.x += 6.0f * kDt;
    out = anchor.update(hand, 0.4f, false, kDt);
  }
  CHECK(out.x > 0.5f);  // dragged, so far
  for (int i = 0; i < 200; ++i) {
    hand.x += 6.0f * kDt;
    out = anchor.update(hand, 3.0f, true, kDt);
    // Back to where it rested: the anchor reaches past the 80 ms of drag.
    CHECK(length(out - rest) < 0.03f);
  }
  CHECK(anchor.holding());

  // The swing is over and the hand has stopped: the bat slides to the hand.
  const Vec3 early = anchor.update(hand, 0.2f, false, kDt);
  CHECK(length(early - rest) < 0.05f);  // no jump on release
  for (int i = 0; i < 600; ++i) out = anchor.update(hand, 0.2f, false, kDt);
  CHECK(length(out - hand) < 0.01f);
}

TEST_CASE("a fast move that never becomes a swing is not held for long", "[anchor]") {
  BatAnchor anchor;
  Vec3 hand{0, 1.0f, 1.5f}, out;
  for (int i = 0; i < 300; ++i) anchor.update(hand, 0.1f, false, kDt);
  // Fast, sustained, and no swing live (the detector is in its recovery).
  for (int i = 0; i < 1000; ++i) {
    hand.x += 0.5f * kDt;
    out = anchor.update(hand, 1.2f, false, kDt);
  }
  CHECK_FALSE(anchor.holding());
  CHECK(length(out - hand) < 0.01f);
}
