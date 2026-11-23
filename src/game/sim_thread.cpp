#include "game/sim_thread.hpp"

#include "concurrency/qos.hpp"
#include "game/hand_control.hpp"
#include "estimation/palm_filter.hpp"
#include "game/paddle_surface.hpp"
#include "game/player_paddle.hpp"
#include "core/signpost.hpp"
#include "game/meeting_point.hpp"
#include "game/opponent_planner.hpp"
#include "game/shot.hpp"
#include "game/swing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>

namespace pp {

namespace {

using Clock = std::chrono::steady_clock;

constexpr real kSimDt = 1.0f / 1000.0f;                    // simulation step
constexpr auto kWakeInterval = std::chrono::microseconds(1000);
constexpr double kMaxCatchUp = 0.05;                       // s: after a stall, do not replay it all

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

// Running statistics over one-second windows.
struct WakeStats {
  double late_sum = 0, late_max = 0, busy = 0, window = 0;
  int wakes = 0;

  void add(double late, double busy_seconds, double elapsed, Snapshot& out) {
    late_sum += late;
    late_max = std::max(late_max, late);
    busy += busy_seconds;
    window += elapsed;
    ++wakes;
    if (window >= 1.0) {
      out.wake_late_mean_us = static_cast<float>(1e6 * late_sum / wakes);
      out.wake_late_max_us = static_cast<float>(1e6 * late_max);
      out.busy_fraction = static_cast<float>(busy / window);
      *this = {};
    }
  }
};

}  // namespace

SimThread::SimThread(double (*host_now)())
    : host_now_(host_now), thread_([this](std::stop_token stop) { run(stop); }) {}

SimThread::~SimThread() = default;  // jthread requests stop and joins

void SimThread::push_hand(const HandObservation& obs) {
  if (!hands_.try_push(obs)) hands_dropped_.fetch_add(1, std::memory_order_relaxed);
}

bool SimThread::send(const SimCommand& command) { return commands_.try_push(command); }

void SimThread::run(std::stop_token stop) {
  name_current_thread("palmrally.sim");
  opponent_planner();  // allocate its batch now, not at the opponent's first stroke
  make_current_thread_interactive();

  Match match;
  PlayerPaddle paddle;
  HandControl hand_control;
  BallCue cue;                  // the incoming ball as of the previous wake-up
  HandControlOutput hand;
  Vec3 contact_shift;           // moves the drawn bat onto the ball while it is struck
  Snapshot snap;
  WakeStats stats;
  TelemetryTracker telemetry;
  MeetingPointTracker meeting;
  real stepped = 0;  // simulation seconds advanced on the previous wake-up
  // Recording must never slow the simulation: if the main thread is not
  // draining the queue, events are dropped (and counted), not waited for.
  const auto record = [&](const TelemetryEvent& event) {
    if (!telemetry_.try_push(event)) ++snap.telemetry_dropped;
  };

  SwingDetector mouse_swing{kMouseSwing};
  SwingRecord swing_record;     // the swing in progress, for the log
  bool swing_open = false;
  double last_swing_start = 0;
  int hits_seen = 0;

  real speed = kDefaultSimSpeed;
  bool use_hand = false;
  Vec3 toss_origin = paddle.pos();
  double owed = 0;  // simulation seconds not yet stepped

  // How long one mouse step usually takes, in wall seconds, learned from the
  // gaps between messages.
  double pointer_gap = 1.0 / 60.0;
  Clock::time_point last_pointer{}, last_wake = Clock::now();

  Clock::time_point next_wake = Clock::now();
  while (!stop.stop_requested()) {
    // Absolute deadlines: sleeping "for 1 ms" each time would let every late
    // wake-up push all later ones back. Sleeping UNTIL a schedule does not
    // accumulate that error.
    next_wake += kWakeInterval;
    std::this_thread::sleep_until(next_wake);
    const Clock::time_point woke = Clock::now();
    const double late = std::max(0.0, seconds(woke - next_wake));
    if (late > kMaxCatchUp) next_wake = woke;  // stalled (debugger, lid closed): start afresh

    const double elapsed = std::min(seconds(woke - last_wake), kMaxCatchUp);
    last_wake = woke;

    // ---- commands from the main thread ----
    SimCommand cmd;
    bool pointer_moved = false;
    Vec3 pointer_target;
    while (commands_.try_pop(cmd)) {
      switch (cmd.kind) {
        case SimCommand::Kind::PointerTarget:
          pointer_target = cmd.target;
          pointer_moved = true;
          break;
        case SimCommand::Kind::Restart: {
          const bool assist = match.assist;
          const int difficulty = match.difficulty;
          match = Match{};
          match.assist = assist;
          match.difficulty = difficulty;
          hits_seen = 0;
          break;
        }
        case SimCommand::Kind::SetAssist: match.assist = cmd.flag; break;
        case SimCommand::Kind::SetDifficulty:
          match.difficulty = std::clamp(static_cast<int>(cmd.value), 0, kDifficultyLevels - 1);
          break;
        case SimCommand::Kind::SetSpeed: speed = std::clamp(cmd.value, 0.40f, 1.00f); break;
        case SimCommand::Kind::UseHand:
          use_hand = cmd.flag;
          hand_control.reset();
          hand = {};
          {
            const bool enabled = paddle.magnet.enabled;
            paddle.magnet = use_hand ? kHandMagnet : kMouseMagnet;
            paddle.magnet.enabled = enabled;
          }
          mouse_swing = SwingDetector{kMouseSwing};
          swing_open = false;
          break;
        case SimCommand::Kind::SetMagnet: paddle.magnet.enabled = cmd.flag; break;
        case SimCommand::Kind::Recentre:
          hand_control.reset();
          break;
      }
    }
    if (pointer_moved) {
      if (last_pointer != Clock::time_point{}) {
        const double gap = std::clamp(seconds(woke - last_pointer), 0.002, 0.050);
        pointer_gap += (gap - pointer_gap) * 0.1;
      }
      last_pointer = woke;
    }

    // ---- observations from the capture thread ----
    HandObservation obs;
    while (hands_.try_pop(obs)) {
      if (use_hand) {
        hand_control.observe(obs);  // every frame goes through the filter, in order
        record({.kind = TelemetryEvent::Kind::Hand, .hand = obs, .chance = {}, .point = {}, .swing = {}});
      }
      snap.hand = obs;
      snap.palm = palm_from(obs);
      ++snap.hands_received;
    }

    // ---- the player's bat ----
    // Camera: the whole chain from observations to bat is in
    // game/hand_control.hpp (the game moves the body, the hand moves the arm,
    // and a swing is judged by the hand's own speed in the image, in any
    // direction). Asked a thousand times a second, so the bat moves smoothly
    // between the 30 camera frames instead of in steps.
    // Mouse: the bat goes where the pointer is and is judged by its own speed.
    const double now = host_now_();
    SwingDetector& swing = use_hand ? hand_control.swing() : mouse_swing;
    bool swing_began = false;
    if (use_hand) {
      hand = hand_control.update(now, static_cast<real>(elapsed), stepped, cue);
      swing_began = hand.swing_began;
      // The simulation runs slowed down, so in its own time the hand is faster.
      // (Camera on but hand not visible this moment: the bat keeps its place.
      // It used to jump to the mouse cursor, which in recorded play happened
      // mid-rally every time a few frames lost the palm.)
      paddle.place(hand.bat, hand.vel / speed);
    } else {
      swing_began = swing.update(paddle.vel() * speed, static_cast<real>(elapsed));
      if (pointer_moved) paddle.step_toward(pointer_target, static_cast<real>(pointer_gap) * speed);
    }
    if (swing_began) {
      ++snap.swings;
      last_swing_start = now;
      swing_open = true;
      swing_record = {.time = now, .hand = use_hand};
      swing_record.ball_due_in = paddle.has_meeting_point() && paddle.time_to_arrival() > -0.05f
                                     ? paddle.time_to_arrival() / speed
                                     : real{-1};
    }
    const BatFace& face = hand_control.face();

    // ---- where will the ball meet the paddle? ----
    // Feeds the on-screen ring, the magnet and the timing cue. The forward
    // simulation behind it only runs when the ball has been deflected (see
    // game/meeting_point.hpp); otherwise this is a countdown.
    if (match.phase == Phase::PreServe) toss_origin = paddle.raw_pos();
    std::optional<Vec3> meeting_now;
    snap.has_ring = false;
    snap.ring_good = true;
    {
      PP_INTERVAL("meeting point");
      // Hand: a fixed vertical plane just behind the table. (It used to
      // follow the hand's estimated depth; when that estimate wandered
      // forward over the table, the plane was reached before the bounce, the
      // ball counted as not hittable, and ring and magnet vanished.)
      // Mouse: the tilted surface the mouse paddle lives on.
      const MeetingPoint meet =
          use_hand ? meeting.update(match, {0, 0, hand_control.filter().config().base_z}, {0, 0, -1}, stepped)
                   : meeting.update(match, kSurfacePoint, surface_toward_ball(), stepped);
      const BallState& ball = match.world.ball;
      if (meet.known) {
        snap.has_ring = true;
        snap.ring = meet.point;
        snap.ring_good = meet.hittable;
      }
      // What the body goes for: a ball the player may hit, and only that
      // (not the toss, which comes down where the bat already is).
      cue = {.incoming = meet.hittable, .meeting_x = meet.point.x, .due_in = meet.time};
      if (meet.hittable) {
        paddle.set_meeting_point(meet.point, meet.time, /*pull_depth=*/true);
        meeting_now = meet.point;
      } else if (match.phase == Phase::Rally && match.stage == Stage::ServeToss &&
                 match.server == Side::Player) {
        // The toss comes straight back down to where it left the paddle.
        const Vec3 drop{ball.pos.x, toss_origin.y, ball.pos.z + 0.035f};
        const real falling_for =
            ball.vel.y < 0 ? std::clamp((ball.pos.y - drop.y) / 2.0f, 0.0f, 0.5f) : 0.5f;
        snap.has_ring = true;
        snap.ring = drop;
        paddle.set_meeting_point(drop, falling_for, /*pull_depth=*/true);
      } else {
        paddle.clear_meeting_point();
      }
    }

    // ---- step the simulation ----
    owed += elapsed * static_cast<double>(speed);
    const int ticks = static_cast<int>(owed / static_cast<double>(kSimDt));
    owed -= ticks * static_cast<double>(kSimDt);
    stepped = static_cast<real>(ticks) * kSimDt;
    PP_INTERVAL("step");
    for (int i = 0; i < ticks; ++i) {
      const Vec3 at = paddle.pos();
      const Vec3 dir = swing.direction();
      const real speed_before = length(match.world.ball.vel);
      const ContactMask hit = tick(match,
                                   {.pos = at,
                                    .vel = paddle.vel(),
                                    .normal = use_hand ? face_normal(face.turn(), face.closed())
                                                       : synthetic_face_normal(at),
                                    .swing_mode = true,
                                    .swinging = swing.live(),
                                    .swing_power = swing.power(),
                                    .swing_dir = dir,
                                    .swing_age = swing.age(),
                                    .face_turn = use_hand ? face.turn() : real{0},
                                    .face_closed = use_hand ? face.closed() : real{0}},
                                   kSimDt);
      paddle.advance(kSimDt, kSimDt / speed);
      if (match.player_hits != hits_seen) {  // the swing connected: it is spent
        hits_seen = match.player_hits;
        swing_record.hit = true;
        swing_record.age_at_hit = swing.age();
        swing.consume();
      }
      if (hit & (contact::kPaddle0 | contact::kPaddle1)) ++snap.paddle_hits;
      if (hit & (contact::kTableNear | contact::kTableFar)) ++snap.table_hits;
      if (hit & contact::kNet) ++snap.net_hits;
      if (hit & contact::kFloor) ++snap.floor_hits;
      if (hit != contact::kNone) snap.impact_speed = speed_before;
      ++snap.tick;
    }

    // ---- record ----
    if (swing_open && !swing.live()) {  // the swing is over, one way or the other
      swing_open = false;
      swing_record.peak_speed = swing.peak_speed();
      swing_record.power = swing.power();
      swing_record.direction = swing.direction();
      record({.kind = TelemetryEvent::Kind::Swing, .hand = {}, .chance = {}, .point = {},
              .swing = swing_record});
    }
    if (const auto chance = telemetry.update(match, paddle,
                                             {.now = now,
                                              .sim_speed = speed,
                                              .hand = use_hand,
                                              .meeting = meeting_now,
                                              .last_swing_start = last_swing_start})) {
      record({.kind = TelemetryEvent::Kind::Chance, .hand = {}, .chance = *chance, .point = {}, .swing = {}});
    }
    if (const auto point = telemetry.point(match, now)) {
      record({.kind = TelemetryEvent::Kind::Point, .hand = {}, .chance = {}, .point = *point, .swing = {}});
    }

    // ---- publish ----
    snap.ball = match.world.ball;
    snap.ball_radius = match.world.ball_params.radius;
    snap.paddles = match.paddles;
    // Drawn where the hand has it: the rules' bat plus the swing's own travel.
    // And at the moment of the hit it is drawn ON the ball. A swing connects
    // when the ball is anywhere within reach (generous on purpose), which
    // left the ball visibly leaving from thin air a hand's breadth from the
    // bat. The bat goes to the ball for the length of the contact and eases
    // back after it.
    const Vec3 drawn_at = paddle.pos() + (use_hand ? hand.display_offset : Vec3{});
    if (match.strike_hold > 0) {
      contact_shift = match.world.ball.pos + Vec3{0, 0, 0.025f} - drawn_at;
    } else {
      contact_shift *= std::exp(-static_cast<real>(elapsed) / 0.08f);
    }
    snap.paddles[index(Side::Player)].pos = drawn_at + contact_shift;
    snap.body_x = use_hand ? hand.body_x : real{0};
    snap.paddles[index(Side::Player)].normal =
        use_hand ? face_normal(face.turn(), face.closed()) : synthetic_face_normal(paddle.pos());
    snap.paddle_roll = use_hand ? face.roll() : real{0};
    snap.paddle_pull = paddle.pull();
    snap.score = match.score;
    snap.server = match.server;
    snap.last_winner = match.last_winner;
    snap.phase = match.phase;
    snap.stage = match.stage;
    snap.last_reason = match.last_reason;
    snap.assist = match.assist;
    snap.magnet = paddle.magnet.enabled;
    snap.speed = speed;
    snap.hand_in_control = use_hand && hand.hand_seen;
    snap.swinging = swing.live();
    snap.swing_age = swing.age();
    snap.swing_live_for = swing.config().live_for;
    snap.swing_power = swing.power();
    snap.ball_due_in = paddle.has_meeting_point() && paddle.time_to_arrival() > 0
                           ? paddle.time_to_arrival() / speed
                           : real{-1};
    snap.player_hits = static_cast<std::uint32_t>(match.player_hits);
    snap.last_stroke = match.last_stroke;
    snap.last_timing = match.last_timing;
    snap.last_target = match.last_target;
    snap.ball_on_bat = match.strike_hold > 0;
    snap.swing_dir = swing.direction();
    snap.last_hit_speed = match.last_hit_speed;
    snap.difficulty = match.difficulty;
    snap.bat_held = use_hand && hand.holding;
    snap.hands_dropped = hands_dropped_.load(std::memory_order_relaxed);
    stats.add(late, seconds(Clock::now() - woke), elapsed, snap);
    snapshots_.publish(snap);
  }
}

}  // namespace pp
