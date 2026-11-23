#pragma once
// What gets recorded while you play, and why.
//
// The aim is to replace guesses with measurements when tuning how the game
// feels. Three kinds of record:
//
//  1. Every camera observation, raw. With these the whole tracking pipeline
//     can be re-run offline with different filter settings and compared on
//     the same hand movements, instead of asking you to wave again.
//
//  2. One "chance" per ball you could have returned. This is where "hard to
//     hit" becomes numbers:
//       - how far the paddle was from the ball when it arrived, before any
//         help (sizes the magnet: its reach should cover most real misses,
//         and no more);
//       - whether that error leans one way (a constant offset means the
//         hand-to-paddle mapping or the delay prediction is off, which no
//         amount of practice fixes);
//       - how long before arrival the paddle got there (reaction time: says
//         whether the ball is too fast or the ring appears too late);
//       - how fast the paddle was moving (sets the swing-speed-to-pace scale
//         from how you really swing);
//       - what happened to the shot (net, long, good).
//
//  3. One record per point: who won it, why, and how long the rally was.
//     The balance of the game and where points are being lost.

#include "core/types.hpp"
#include "game/match.hpp"
#include "game/player_paddle.hpp"
#include "math/vec3.hpp"
#include "platform/hand_tracker.hpp"

#include <cstdint>
#include <optional>

namespace pp {

enum class ShotResult : std::uint8_t { None, Good, Net, Out, OwnSide };
const char* describe(ShotResult r);

struct ChanceRecord {
  double time = 0;            // host clock, when the ball arrived
  bool hit = false;
  bool hand = false;          // hand tracking in control (else mouse)
  bool magnet = false;
  bool assist = false;
  real sim_speed = 1;

  Vec3 meeting;               // where ball and paddle plane met
  Vec3 raw;                   // paddle from input alone, at arrival
  Vec3 paddle;                // paddle actually used (magnet included)
  real miss_raw = 0;          // distance from `raw` to `meeting`, in the paddle plane
  real pull = 0;              // magnet strength at arrival
  real notice = 0;            // s (real time) from the ball being struck to arrival
  real settled_for = -1;      // s (real time) the paddle was within 15 cm before arrival; -1 never

  real ball_speed = 0;        // m/s, incoming
  Vec3 paddle_vel;            // m/s (real time) at arrival
  ShotResult result = ShotResult::None;

  // The swing nearest this ball: how long before the ball arrived it began
  // (s of real time; negative = began after the ball had arrived). kNoSwing
  // if there was none within a second and a half. This is what separates the
  // three ways to miss: did not swing, swung at the wrong time, or swung in
  // time with the bat out of reach.
  static constexpr real kNoSwing = -99;
  real swing_lead = kNoSwing;
};

// One record per swing, whether or not it hit anything. Sets the detector's
// thresholds: how fast real swings are, and how they are timed.
struct SwingRecord {
  double time = 0;            // host clock, when the swing began
  bool hand = false;
  bool hit = false;
  real peak_speed = 0;        // image widths/s (hand) or m/s (mouse)
  real power = 0;             // 0..1 as the game scored it
  Vec3 direction;             // unit vector at peak speed
  real ball_due_in = -1;      // s (real) until the ball was due when the swing began; -1 no ball
  real age_at_hit = -1;       // s (real) from swing start to contact; -1 if it missed
};

struct PointRecord {
  double time = 0;
  Side winner = Side::Player;
  Side server = Side::Player;
  PointReason reason = PointReason::None;
  int rally_hits = 0;
  int score_player = 0, score_opponent = 0;
};

// One slot in the sim -> main telemetry queue. Plain data, so it can go
// through the lock-free ring; `kind` says which member is meaningful.
struct TelemetryEvent {
  enum class Kind : std::uint8_t { Hand, Chance, Point, Swing };
  Kind kind = Kind::Hand;
  HandObservation hand;
  ChanceRecord chance;
  PointRecord point;
  SwingRecord swing;
};

// Watches the match from the simulation thread and produces the chance and
// point records. Call update() once per wake-up, after stepping.
class TelemetryTracker {
 public:
  struct Inputs {
    double now = 0;           // host clock
    real sim_speed = 1;
    bool hand = false;
    // The latest meeting-point prediction for the incoming ball, if one was
    // made this wake-up and the ball will be legal to hit.
    std::optional<Vec3> meeting;
    double last_swing_start = 0;  // host clock; 0 if there has been none
  };

  // Each returns a record at most once per event.
  std::optional<ChanceRecord> update(const Match& match, const PlayerPaddle& paddle,
                                     const Inputs& in);
  std::optional<PointRecord> point(const Match& match, double now);

 private:
  enum class State : std::uint8_t { Idle, Incoming, AwaitingResult };
  State state_ = State::Idle;
  ChanceRecord record_;
  bool has_meeting_ = false;
  bool passed_ = false;       // the ball has gone by the paddle
  double struck_at_ = 0, within_since_ = -1, arrived_at_ = 0;
  Phase last_phase_ = Phase::PreServe;
  real last_ball_speed_ = 0;

  void sample(const Match& match, const PlayerPaddle& paddle, const Inputs& in);
};

}  // namespace pp
