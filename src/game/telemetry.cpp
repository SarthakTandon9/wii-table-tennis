#include "game/telemetry.hpp"

#include <cmath>

namespace pp {

namespace {

constexpr real kSettledRadius = 0.15f;  // "the paddle is there": within this of the meeting point

bool ball_incoming(const Match& m) {
  return m.phase == Phase::Rally && m.striker == Side::Opponent && m.stage != Stage::ServeToss;
}

}  // namespace

const char* describe(ShotResult r) {
  switch (r) {
    case ShotResult::None: return "none";
    case ShotResult::Good: return "good";
    case ShotResult::Net: return "net";
    case ShotResult::Out: return "out";
    case ShotResult::OwnSide: return "own_side";
  }
  return "";
}

// Refreshes the "at arrival" fields. Called every wake-up while the ball is
// coming, so that whenever it arrives the record holds the latest values.
void TelemetryTracker::sample(const Match& match, const PlayerPaddle& paddle, const Inputs& in) {
  if (in.meeting) {
    record_.meeting = *in.meeting;
    has_meeting_ = true;
  }
  if (!has_meeting_) return;

  record_.raw = paddle.raw_pos();
  record_.paddle = paddle.pos();
  record_.pull = paddle.pull();
  // Paddle velocity is kept in simulation time; report it as the hand moved.
  record_.paddle_vel = paddle.vel() * in.sim_speed;
  record_.ball_speed = last_ball_speed_;
  record_.hand = in.hand;
  record_.magnet = paddle.magnet.enabled;
  record_.assist = match.assist;
  record_.sim_speed = in.sim_speed;

  Vec3 gap = record_.meeting - record_.raw;
  gap.z = 0;
  record_.miss_raw = length(gap);
  if (record_.miss_raw <= kSettledRadius) {
    if (within_since_ < 0) within_since_ = in.now;
  } else {
    within_since_ = -1;
  }
}

std::optional<ChanceRecord> TelemetryTracker::update(const Match& match,
                                                     const PlayerPaddle& paddle,
                                                     const Inputs& in) {
  std::optional<ChanceRecord> done;
  const auto finish = [&](ShotResult result) {
    record_.result = result;
    done = record_;
    state_ = State::Idle;
  };

  if (state_ == State::AwaitingResult) {
    // The player hit it. What became of the shot?
    if (match.phase != Phase::Rally) {
      const bool lost = match.last_winner == Side::Opponent;
      finish(!lost ? ShotResult::Good
             : match.last_reason == PointReason::Net ? ShotResult::Net
             : match.last_reason == PointReason::OwnSide ? ShotResult::OwnSide
                                                          : ShotResult::Out);
    } else if (match.stage == Stage::AwaitHit || match.striker == Side::Opponent) {
      finish(ShotResult::Good);  // it landed on their half
    }
  }

  if (state_ == State::Idle && ball_incoming(match)) {
    state_ = State::Incoming;
    record_ = {};
    has_meeting_ = false;
    passed_ = false;
    within_since_ = -1;
    struck_at_ = in.now;
  }

  if (state_ == State::Incoming) {
    const auto close_out = [&](bool hit) {
      const double arrival = hit ? in.now : arrived_at_;
      record_.hit = hit;
      record_.time = arrival;
      record_.notice = static_cast<real>(arrival - struck_at_);
      const double lead = arrival - in.last_swing_start;
      record_.swing_lead = in.last_swing_start > 0 && std::abs(lead) < 1.5
                               ? static_cast<real>(lead)
                               : ChanceRecord::kNoSwing;
      record_.settled_for = within_since_ >= 0 ? static_cast<real>(arrival - within_since_) : -1;
    };
    if (ball_incoming(match)) {
      // A missed ball keeps flying for a while before the point is called.
      // Freeze the record at the moment it passed the paddle, so "where was
      // the paddle" means at arrival and not half a second later.
      if (!passed_ && has_meeting_ && match.world.ball.pos.z > record_.meeting.z + 0.03f) {
        passed_ = true;
        arrived_at_ = in.now;
      }
      if (!passed_) {
        sample(match, paddle, in);
        last_ball_speed_ = length(match.world.ball.vel);
      }
    } else if (match.phase == Phase::Rally && match.striker == Side::Player) {
      close_out(true);
      state_ = State::AwaitingResult;
    } else {
      // The point ended with the ball never returned. Only a ball that was
      // returnable counts as a chance; one that flew long or hit the net
      // was the opponent's error.
      if (has_meeting_) {
        if (!passed_) arrived_at_ = in.now;
        close_out(false);
        finish(ShotResult::None);
      } else {
        state_ = State::Idle;
      }
    }
  }
  return done;
}

std::optional<PointRecord> TelemetryTracker::point(const Match& match, double now) {
  const Phase previous = last_phase_;
  last_phase_ = match.phase;
  const bool just_ended = previous == Phase::Rally && match.phase != Phase::Rally;
  if (!just_ended || match.last_reason == PointReason::None ||
      match.last_reason == PointReason::Let) {
    return std::nullopt;
  }
  return PointRecord{.time = now,
                     .winner = match.last_winner,
                     .server = match.server,
                     .reason = match.last_reason,
                     .rally_hits = match.rally_hits,
                     .score_player = match.score[index(Side::Player)],
                     .score_opponent = match.score[index(Side::Opponent)]};
}

}  // namespace pp
