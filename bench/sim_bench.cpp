// How much does the simulation thread's work cost, and where does it go?
//
// Runs exactly what the simulation thread does on each wake-up (rules,
// physics, opponent, meeting-point prediction, paddle, hand filter, swing
// detection, telemetry), but flat out instead of once a millisecond, so the
// cost per wake-up can be measured and a profiler has something to chew on.
//
//   sim_bench [simulated seconds] [periodic|event]
//
// To see where the time goes without Instruments:
//   ./build/release/sim_bench 600 &  sample $! 5
#include "estimation/palm_filter.hpp"
#include "game/match.hpp"
#include "game/meeting_point.hpp"
#include "game/paddle_surface.hpp"
#include "game/player_paddle.hpp"
#include "game/swing.hpp"
#include "game/telemetry.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>

int main(int argc, char** argv) {
  using namespace pp;
  const double simulate_seconds = argc > 1 ? std::atof(argv[1]) : 120.0;
  constexpr real kDt = 1e-3f;

  Match match{Side::Opponent, 0x1234u};
  PlayerPaddle paddle;
  PalmFilter filter;
  SwingDetector swing{kHandSwing};
  TelemetryTracker telemetry;
  const bool periodic = argc > 2 && std::string_view{argv[2]} == "periodic";
  MeetingPointTracker meeting_tracker{periodic ? MeetingPointTracker::Policy::Periodic
                                               : MeetingPointTracker::Policy::OnEvent};
  double now = 50.0, last_swing_start = 0;
  int hits_seen = 0, strokes = 0, points = 0;
  std::uint32_t rng = 0xBEEFu;
  const auto unit = [&] {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return static_cast<float>(rng >> 8) / 16777216.0f;
  };

  const long wakes = static_cast<long>(simulate_seconds * 1000.0);
  const auto start = std::chrono::steady_clock::now();
  for (long i = 0; i < wakes; ++i) {
    if (match.phase == Phase::GameOver) {
      match = Match{Side::Opponent, rng};
      hits_seen = 0;
    }
    if (i % 33 == 0) {  // a camera frame
      HandObservation obs;
      obs.found = true;
      obs.aspect = 4.0f / 3.0f;
      obs.capture_time = now - 0.06;
      const float cx = 0.5f + 0.1f * std::sin(static_cast<float>(now)) + (unit() - 0.5f) * 0.01f;
      for (JointPoint& j : obs.joints) j = {.x = cx, .y = 0.3f, .confidence = 0.9f};
      obs.joints[0].y = 0.22f;
      obs.joints[5].x = cx - 0.03f;
      obs.joints[17].x = cx + 0.03f;
      filter.observe(obs);
    }
    const HandPose hand = filter.pose_at(now);
    if (hand.valid) paddle.place(hand.pos, hand.vel);

    const MeetingPoint meet = meeting_tracker.update(match, {0, 0, 1.50f}, {0, 0, -1}, kDt);
    if (meet.hittable) paddle.set_meeting_point(meet.point, meet.time, true);
    else paddle.clear_meeting_point();

    const bool due = paddle.has_meeting_point() && paddle.time_to_arrival() < 0.15f;
    const bool serving = match.stage == Stage::ServeToss && match.phase == Phase::Rally &&
                         match.world.ball.vel.y < 0;
    if (swing.update({(due || serving) ? 2.5f : 0.1f, 0.5f, 0}, kDt)) last_swing_start = now;

    const Vec3 at = paddle.pos();
    const Vec3 dir = swing.direction();
    tick(match,
         {.pos = at, .vel = paddle.vel(), .normal = synthetic_face_normal(at), .swing_mode = true,
          .swinging = swing.live(), .swing_power = swing.power(), .swing_dir = dir,
          .swing_age = swing.age()},
         kDt);
    paddle.advance(kDt, kDt);
    if (match.player_hits != hits_seen) {
      hits_seen = match.player_hits;
      ++strokes;
      swing.consume();
    }
    now += static_cast<double>(kDt);
    telemetry.update(match, paddle,
                     {.now = now, .sim_speed = 1, .hand = hand.valid,
                      .meeting = meet.hittable ? std::optional<Vec3>{meet.point} : std::nullopt,
                      .last_swing_start = last_swing_start});
    if (telemetry.point(match, now)) ++points;
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  std::printf("[%s prediction] %.0f simulated seconds (%d points, %d strokes) in %.3f s\n",
              periodic ? "periodic" : "on-event", simulate_seconds, points, strokes, elapsed);
  std::printf("  %.2f us per wake-up: %.2f%% of the 1 ms budget, %.0fx faster than real time\n",
              1e6 * elapsed / static_cast<double>(wakes),
              100.0 * 1e3 * elapsed / static_cast<double>(wakes), simulate_seconds / elapsed);
  std::printf("  meeting-point predictions run: %ld (%.1f per simulated second)\n",
              meeting_tracker.predictions(),
              static_cast<double>(meeting_tracker.predictions()) / simulate_seconds);
  return 0;
}
