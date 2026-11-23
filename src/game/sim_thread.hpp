#pragma once
// The simulation, on its own thread.
//
//   capture thread --[SpscRing<HandObservation>]--\
//                                                  >--> SIM THREAD --[TripleBuffer<Snapshot>]--> main
//   main thread ----[SpscRing<SimCommand>]--------/
//
// Three threads, and nothing is shared between them except these three
// lock-free channels. The simulation never waits for the camera (30 Hz) or
// for the renderer (display rate); it ticks at 1 kHz regardless. That is the
// point of the design: a slow frame or a dropped camera frame cannot make the
// ball stutter or tunnel.
//
// Why two different kinds of channel:
//  - Into the sim, every message matters (a restart command must not be
//    lost), so those are queues.
//  - Out of the sim, only the newest state matters, so that is a triple
//    buffer: the renderer always gets the latest and never a backlog.

#include "concurrency/spsc_ring.hpp"
#include "concurrency/triple_buffer.hpp"
#include "core/types.hpp"
#include "estimation/palm.hpp"
#include "game/match.hpp"
#include "game/telemetry.hpp"
#include "math/vec3.hpp"
#include "platform/hand_tracker.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <stop_token>
#include <thread>

namespace pp {

// Main thread -> sim.
struct SimCommand {
  enum class Kind : std::uint8_t {
    PointerTarget,  // `target`: where the mouse wants the paddle
    Restart,
    SetAssist,      // `flag`
    SetSpeed,       // `value`: simulation speed, 1 = real time
    UseHand,        // `flag`: let camera observations drive the paddle
    SetMagnet,      // `flag`: pull the paddle toward the ball's meeting point
    Recentre,       // make the hand's current position the paddle's rest position
    SetDifficulty,  // `value`: opponent level, 0 .. kDifficultyLevels - 1
  };
  Kind kind = Kind::PointerTarget;
  bool flag = false;
  real value = 0;
  Vec3 target;
};

// Sim -> main: everything needed to draw one frame and nothing else. Plain
// data, copied whole, so the renderer never touches live simulation state.
// The simulation runs a little slower than real time: the camera sees the
// hand 55 ms late and a swing takes two more frames to register, which at
// full speed is most of the time a ball spends near the bat. It started at
// 70% while returning the ball was the problem; with 94% of balls returned in
// recorded play there is room to bring it closer to the real thing.
inline constexpr real kDefaultSimSpeed = 0.80f;

struct Snapshot {
  // Scene.
  BallState ball;
  real ball_radius = 0.02f;
  std::array<Paddle, 2> paddles;
  bool has_ring = false;   // where to put the paddle
  bool ring_good = true;   // false: the ball is going long, leave it
  Vec3 ring;

  // Game state.
  std::array<int, 2> score{};
  Side server = Side::Player;
  Side last_winner = Side::Player;
  Phase phase = Phase::PreServe;
  Stage stage = Stage::ServeToss;
  PointReason last_reason = PointReason::None;
  bool assist = true;
  bool magnet = true;
  real speed = kDefaultSimSpeed;
  real paddle_roll = 0;    // radians: how the hand is rotated, for drawing the bat
  real paddle_pull = 0;    // 0..1: how strongly the magnet is acting right now

  // Swing.
  bool swinging = false;        // a swing is live right now
  real swing_age = 0;           // s (real) since it began
  real swing_live_for = 0.4f;   // how long a swing stays live
  real swing_power = 0;         // 0..1
  real ball_due_in = -1;        // s (real) until the ball reaches the bat; -1 none coming
  std::uint32_t swings = 0;     // swings begun since the thread started
  std::uint32_t player_hits = 0;
  StrokeKind last_stroke = StrokeKind::Drive;  // what the latest hit was read as
  StrokeTiming last_timing = StrokeTiming::Good;
  Vec3 last_target;             // where on the far half it was sent
  bool ball_on_bat = false;     // a swing has connected and the ball is being struck
  Vec3 swing_dir;               // which way the hand was going at the swing's peak
  real last_hit_speed = 0;      // m/s the player's latest stroke left the bat at
  int difficulty = kDefaultDifficulty;
  bool bat_held = false;        // the rules' bat is holding its place while the hand swings
  real body_x = 0;              // where the player stands, sideways (the view follows it)

  // Events since the thread started. The renderer plays a sound whenever one
  // of these moves. Counters survive "latest wins": a snapshot can be skipped
  // without the event being lost.
  std::uint32_t paddle_hits = 0, table_hits = 0, net_hits = 0, floor_hits = 0;
  real impact_speed = 0;  // m/s: how fast the ball was going into the latest of those

  // Camera.
  HandObservation hand;      // newest observation the sim has consumed
  Palm palm;
  bool hand_in_control = false;
  std::uint64_t hands_received = 0;
  std::uint64_t hands_dropped = 0;  // capture thread found the queue full
  std::uint64_t telemetry_dropped = 0;  // main thread was not draining the log queue

  // Timing, over the last second.
  std::uint64_t tick = 0;           // simulation ticks so far
  float wake_late_mean_us = 0;      // how late the 1 kHz wake-ups arrive
  float wake_late_max_us = 0;
  float busy_fraction = 0;          // share of wall time spent computing
};

class SimThread {
 public:
  // `host_now` returns seconds on the clock that HandObservation timestamps
  // use (HandTracker::now). Passed in so this file stays free of Apple APIs.
  explicit SimThread(double (*host_now)());
  ~SimThread();  // stops and joins
  SimThread(const SimThread&) = delete;
  SimThread& operator=(const SimThread&) = delete;

  // Capture thread only. Never blocks; if the sim has fallen behind, the
  // observation is dropped and counted.
  void push_hand(const HandObservation& obs);

  // Main thread only. Returns false if the command queue is full.
  bool send(const SimCommand& command);

  // Main thread only. The newest state; valid until the next call.
  const Snapshot& latest() { return snapshots_.latest(); }

  // Main thread only. Recorded events for the session log, oldest first.
  // Returns false when there are none waiting.
  bool poll_telemetry(TelemetryEvent& out) { return telemetry_.try_pop(out); }

 private:
  void run(std::stop_token stop);

  double (*host_now_)();
  SpscRing<HandObservation, 64> hands_;
  SpscRing<SimCommand, 256> commands_;
  TripleBuffer<Snapshot> snapshots_;
  SpscRing<TelemetryEvent, 512> telemetry_;  // sim -> main, for the session log
  std::atomic<std::uint64_t> hands_dropped_{0};
  std::jthread thread_;  // last member: starts after, and joins before, the rest
};

}  // namespace pp
