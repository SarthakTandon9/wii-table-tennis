#include "game/hand_control.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace pp;

namespace {

constexpr double kFrame = 1.0 / 30.0;
constexpr double kDelay = 0.060;  // camera exposure to the game receiving the frame

HandObservation hand_at(double t, float x, float y) {
  HandObservation obs;
  obs.found = true;
  obs.aspect = 4.0f / 3.0f;
  obs.capture_time = t;
  obs.detect_time = t + kDelay;
  for (JointPoint& j : obs.joints) j = {.x = x, .y = y, .confidence = 0.9f};
  obs.joints[static_cast<std::size_t>(HandJoint::Wrist)].y = y - 0.10f;
  obs.joints[static_cast<std::size_t>(HandJoint::IndexMCP)].x = x - 0.03f;
  obs.joints[static_cast<std::size_t>(HandJoint::RingMCP)].x = x + 0.015f;
  obs.joints[static_cast<std::size_t>(HandJoint::LittleMCP)].x = x + 0.03f;
  return obs;
}

// Runs the control for `seconds` with the hand following `path(t)`, as the
// game does: frames arrive kDelay late, the control is asked every millisecond.
double g_next_frame = 0;  // capture time of the next camera frame (reset by each test)

template <typename Path>
HandControlOutput run(HandControl& control, double& t, double seconds, Path path, BallCue cue = {}) {
  HandControlOutput out;
  if (t == 0) g_next_frame = 0;
  double& next_frame = g_next_frame;
  const double end = t + seconds;
  for (; t < end; t += 0.001) {
    while (next_frame + kDelay <= t) {
      const auto [x, y] = path(next_frame);
      control.observe(hand_at(next_frame, x, y));
      next_frame += kFrame;
    }
    BallCue now = cue;
    if (cue.incoming) now.due_in = cue.due_in - static_cast<real>(seconds - (end - t));
    out = control.update(t, 0.001f, 0.001f, now);
  }
  return out;
}

const auto still = [](double) { return std::pair{0.60f, 0.30f}; };

}  // namespace

TEST_CASE("the body goes to the ball by itself and comes back afterwards", "[hand]") {
  HandControl control;
  double t = 0;
  run(control, t, 1.0, still);
  // A ball due 0.6 s from now, 60 cm to the right. The hand does not move.
  const HandControlOutput there =
      run(control, t, 0.6, still, {.incoming = true, .meeting_x = 0.60f, .due_in = 0.6f});
  CHECK(std::abs(there.body_x - 0.60f) < 0.05f);
  CHECK(std::abs(there.bat.x - 0.60f) < 0.08f);  // and the bat with it

  // The ball has gone: back to the middle.
  const HandControlOutput back = run(control, t, 1.5, still);
  CHECK(std::abs(back.body_x) < 0.05f);
}

TEST_CASE("a ball too far for the time available is not reached", "[hand]") {
  HandControl control;
  double t = 0;
  run(control, t, 1.0, still);
  // 90 cm away, a quarter of a second's warning.
  const HandControlOutput late =
      run(control, t, 0.25, still, {.incoming = true, .meeting_x = 0.90f, .due_in = 0.25f});
  CHECK(late.body_x < 0.55f);
}

TEST_CASE("the hand moves the bat within arm's reach, and no further", "[hand]") {
  HandControl control;
  double t = 0;
  run(control, t, 1.0, still);
  // An unhurried move to the right, then well beyond the learned range.
  const HandControlOutput moved = run(control, t, 1.0, [](double time) {
    return std::pair{0.60f + 0.08f * static_cast<float>(std::min(time - 1.0, 0.5) / 0.5), 0.30f};
  });
  const real arm = moved.bat.x - moved.body_x;
  CHECK(arm > 0.08f);                // the bat went right, relative to the body
  CHECK(arm <= kArmReach + 1e-4f);
  const HandControlOutput far = run(control, t, 1.0, [](double time) {
    return std::pair{0.68f + 0.30f * static_cast<float>(std::min(time - 2.0, 0.6) / 0.6), 0.30f};
  });
  CHECK(far.bat.x - far.body_x <= kArmReach + 1e-4f);
}

TEST_CASE("a swing: the rules' bat holds still, the bat on screen goes with the hand", "[hand]") {
  HandControl control;
  double t = 0;
  const HandControlOutput rest = run(control, t, 1.5, still);

  // The hand whips 30% of the picture to the left in 0.1 s: 3 widths a second, a real swing.
  bool began = false;
  real furthest_shown = 0, furthest_ruled = 0, lunge = 0;
  const auto swing_path = [](double time) {
    return std::pair{0.60f - 0.30f * static_cast<float>(std::clamp((time - 1.5) / 0.1, 0.0, 1.0)), 0.30f};
  };
  for (int step = 0; step < 30; ++step) {
    const HandControlOutput out = run(control, t, 0.01, swing_path);
    began = began || control.swing().live();
    furthest_ruled = std::max(furthest_ruled, std::abs(out.bat.x - rest.bat.x));
    furthest_shown = std::max(furthest_shown, std::abs(out.bat.x + out.display_offset.x - rest.bat.x));
    lunge = std::min(lunge, out.display_offset.z);
  }
  CHECK(began);
  CHECK(furthest_ruled < 0.06f);   // for the rules it stayed put (the anchor reaches back)
  CHECK(furthest_shown > 0.12f);   // on screen it swung with the hand
  CHECK(lunge < -0.05f);           // and went forward, toward the table

  // Afterwards the two are one bat again.
  const HandControlOutput after = run(control, t, 1.2, swing_path);
  CHECK(length(after.display_offset) < 0.01f);
}
