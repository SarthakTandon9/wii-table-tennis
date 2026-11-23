// Proves the simulation loop does not allocate memory.
//
// Why it matters: malloc takes a lock and, now and then, asks the operating
// system for more memory, which can take milliseconds. A loop that must wake
// a thousand times a second cannot afford an occasional multi-millisecond
// call, and "occasional" is exactly the kind of problem that never shows up
// in a quick test and always shows up in a long session.
//
// How: this program replaces the global operator new with one that counts.
// It then runs everything the simulation thread runs per wake-up (rules,
// physics, opponent, shot planning, paddle, hand filter, swing detection,
// telemetry) through many complete points, and requires the count taken
// during that stretch to be zero.
//
// It is its own executable because replacing operator new affects the whole
// program, and the test framework itself allocates freely.
#include "estimation/palm_filter.hpp"
#include "game/match.hpp"
#include "game/opponent_planner.hpp"
#include "game/paddle_surface.hpp"
#include "game/player_paddle.hpp"
#include "game/shot.hpp"
#include "game/swing.hpp"
#include "game/telemetry.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>

namespace {
std::atomic<long> g_allocations{0};
std::atomic<bool> g_counting{false};
}  // namespace

void* operator new(std::size_t size) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1);
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc{};
}
void* operator new(std::size_t size, std::align_val_t align) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1);
  void* p = nullptr;
  if (posix_memalign(&p, static_cast<std::size_t>(align), size == 0 ? 1 : size) == 0) return p;
  throw std::bad_alloc{};
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

int main() {
  using namespace pp;
  constexpr real kDt = 1e-3f;

  Match match{Side::Opponent, 0x1234u};
  PlayerPaddle paddle;
  PalmFilter filter;
  SwingDetector swing{kHandSwing};
  TelemetryTracker telemetry;
  double now = 50.0, last_swing_start = 0;
  int hits_seen = 0, strokes = 0, points = 0, chances = 0, hands = 0;
  std::uint32_t rng = 0xBEEFu;
  const auto unit = [&] {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return static_cast<float>(rng >> 8) / 16777216.0f;
  };

  opponent_planner();  // set up before the loop, as the simulation thread does
  g_counting = true;
  for (int i = 0; i < 400'000 && points < 40; ++i) {
    if (match.phase == Phase::GameOver) {
      match = Match{Side::Opponent, rng};
      hits_seen = 0;  // the new match counts its strokes from zero
    }

    // A camera frame 30 times a second: a hand that wanders and trembles.
    if (i % 33 == 0) {
      HandObservation obs;
      obs.found = true;
      obs.aspect = 4.0f / 3.0f;
      obs.capture_time = now - 0.06;
      const float cx = 0.5f + 0.1f * std::sin(static_cast<float>(now)) + (unit() - 0.5f) * 0.01f;
      for (JointPoint& j : obs.joints) j = {.x = cx, .y = 0.3f, .confidence = 0.9f};
      obs.joints[0].y = 0.22f;
      filter.observe(obs);
      ++hands;
    }
    const HandPose hand = filter.pose_at(now);

    // Meeting point, as the simulation thread computes it.
    std::optional<Vec3> meeting;
    const bool incoming = match.phase == Phase::Rally && match.striker == Side::Opponent;
    if (i % 4 == 0 && incoming) {
      const auto c = predict_crossing(match.world, {0, 0, 1.50f}, {0, 0, -1},
                                      match.stage == Stage::AwaitHit);
      if (c && c->bounced) {
        meeting = c->point;
        paddle.place(c->point + Vec3{0.1f, 0.05f, 0}, hand.vel);
        paddle.set_meeting_point(c->point, c->time);
      } else {
        paddle.clear_meeting_point();
      }
    }

    // Swing whenever a ball is nearly due (or we are serving), so that
    // strokes, serves and the planners behind them all get exercised.
    const bool due = paddle.has_meeting_point() && paddle.time_to_arrival() < 0.15f;
    const bool serving = match.stage == Stage::ServeToss && match.phase == Phase::Rally;
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

    if (telemetry.update(match, paddle, {.now = now, .sim_speed = 1, .hand = hand.valid,
                                         .meeting = meeting,
                                         .last_swing_start = last_swing_start})) {
      ++chances;
    }
    if (telemetry.point(match, now)) ++points;
  }
  g_counting = false;

  const long allocations = g_allocations.load();
  std::printf("%d points, %d returnable balls, %d strokes by the player, %d camera frames\n",
              points, chances, strokes, hands);
  std::printf("allocations in the simulation loop: %ld\n", allocations);
  if (points < 25 || strokes < 80) {
    std::printf("FAILED: the run did not exercise enough of the game to be meaningful\n");
    return 1;
  }
  if (allocations != 0) {
    std::printf("FAILED: the hot path allocated\n");
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
