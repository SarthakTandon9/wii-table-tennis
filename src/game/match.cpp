#include "game/match.hpp"

#include "game/opponent_planner.hpp"
#include "game/shot.hpp"

#include <algorithm>
#include <cmath>

namespace pp {

namespace {

constexpr real kOpponentZ = -1.60f;
// How fast the opponent gets its bat across, and how long it takes to read a
// shot. At 3.6 m/s and 0.12 s it reached every ball: simulated over thousands
// of points, not one was won by hitting past it, so pace and placement meant
// nothing and points ended only on its unforced errors. At these values a
// hard ball to the far corner gets by; a soft one to the middle never does.
// The numbers themselves are per difficulty level: see kOpponentSkill.
const OpponentSkill& skill(const Match& m) {
  return kOpponentSkill[std::clamp(m.difficulty, 0, kDifficultyLevels - 1)];
}
constexpr real kHitCooldown = 0.15f;
constexpr real kQuietLimit = 3.0f;         // a ball nobody touches for this long is dead
constexpr real kTossSpeed = 2.5f;          // rises about 32 cm (law: at least 16 cm)

// xorshift32: tiny, fast, identical on every platform.
real next_unit(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<real>(state >> 8) * (1.0f / 16777216.0f);  // [0, 1)
}

real next_range(std::uint32_t& state, real lo, real hi) {
  return lo + (hi - lo) * next_unit(state);
}

Vec3 opponent_ready(const Match& m) {
  return {0, m.world.table.height + 0.22f, kOpponentZ};
}

bool over_table(const Match& m, Vec3 p) {
  return std::abs(p.x) <= m.world.table.width * 0.5f &&
         std::abs(p.z) <= m.world.table.length * 0.5f;
}

// ---------------------------------------------------------------- scoring

void begin_serve(Match& m) {
  m.phase = Phase::PreServe;
  m.stage = Stage::ServeToss;
  m.striker = m.server;
  m.timer = 1.1f;
  m.rally_hits = 0;
  m.serve_touched_net = false;
  m.world.ball.vel = {};
  m.world.ball.spin = {};
}

void award(Match& m, Side winner, PointReason why) {
  ++m.score[index(winner)];
  m.last_winner = winner;
  m.last_reason = why;
  m.phase = Phase::PointOver;
  m.timer = 1.5f;
}

void replay(Match& m, PointReason why) {  // let, or a missed toss
  m.last_reason = why;
  m.phase = Phase::PointOver;
  m.timer = why == PointReason::Let ? 1.0f : 0.5f;
}

void after_point(Match& m) {
  const int a = m.score[0], b = m.score[1];
  if (game_won(a, b) || game_won(b, a)) {
    m.phase = Phase::GameOver;
    return;
  }
  m.server = server_for(a + b, m.first_server);
  begin_serve(m);
}

// ------------------------------------------------------------------- strokes

// The player's stroke, as struck, decides direction and pace. The assist only
// re-lofts it so it lands: harder hits go deeper and flatter.
// A swing turned into a shot. The hand cannot be read precisely enough to
// simulate a real stroke, so the swing is read for its intent instead:
//   how fast            -> pace, and how deep the ball lands
//   sideways motion     -> where across the table it goes
//   up or down          -> topspin (dips, kicks on) or backspin (floats)
//   early or late       -> pulled one way or pushed the other, as in real play
// and the shot is then planned so that it clears the net and lands.
void assist_swing_stroke(Match& m) {
  BallState& ball = m.world.ball;
  const PaddleInput& in = m.input;
  const real half_l = m.world.table.length * 0.5f, half_w = m.world.table.width * 0.5f;

  // Which way the hand went decides the stroke. Mostly upward is a topspin
  // brush, mostly downward a chop, anything else (across, or a push toward
  // the screen) a drive. "Mostly" is a 40-degree cone, wide enough that a
  // swing meant as vertical is read as one.
  const real up = std::max(in.swing_dir.y, real{0}), down = std::max(-in.swing_dir.y, real{0});
  m.last_stroke = up > 0.64f ? StrokeKind::Topspin : down > 0.64f ? StrokeKind::Chop
                                                                    : StrokeKind::Drive;

  // When the swing met the ball. A swing begun well ahead has passed its
  // best by the time the ball arrives; one that only catches the ball after
  // it has gone by the bat is late. Either still returns the ball, but
  // weaker and less surely, and the game says which it was, so that timing
  // is something the player can feel and correct.
  m.last_timing = m.strike_gap_z > 0.12f     ? StrokeTiming::Late
                  : m.strike_swing_age > 0.25f ? StrokeTiming::Early
                                               : StrokeTiming::Good;
  const bool mistimed = m.last_timing != StrokeTiming::Good;

  // Any swing that connects is a proper hit; power adds to that. (A floor,
  // because a shot at the very bottom of the range is a slow lob that the
  // player did not ask for.)
  const real power = 0.15f + 0.85f * in.swing_power;
  // A chop cannot be hit as hard as a drive: with backspin the ball floats,
  // and at drive pace it would sail off the end.
  const real speed = (5.0f + 6.0f * power) * (1.0f - 0.35f * down) * (mistimed ? 0.82f : 1.0f);
  const real target_z = -(0.45f + 0.43f * power) * half_l * (mistimed ? 0.90f : 1.0f);

  // Where it goes: where the swing went. Swing to the right and the ball
  // goes right; swing straight (up, down, or a push) and it goes down the
  // middle. Nothing else has a say.
  //
  // It was not always so. The aim used to add three more things: the angle
  // of the bat's face (read from how the hand leans), how early the swing
  // was, and a share of where the ball happened to be. Replaying a recorded
  // session of 55 hits through that formula, the swing's own direction
  // accounted for 36% of where the ball was sent. The face term was nearly
  // as large (spread 0.26 against 0.33) and, being a lean of the hand nobody
  // was aiming with, as often against the swing as with it; four hits in ten
  // came at the swing's onset, which the timing term turned into a constant
  // push to the right. To the player that is a shot with a mind of its own.
  const real aim = std::clamp(0.92f * in.swing_dir.x, -1.0f, 1.0f);
  // A full sideways swing reaches 18 cm inside the side line: wide enough to
  // make the opponent move, not a winner by itself. (Aimed at the line, it
  // was: simulated, four points in ten at the middle level ended that way.)
  const real limit = half_w - 0.18f;
  // The same swing sends the ball to the same place, within a hand's width.
  // Only two things loosen that: going flat out (the last sixth of the power
  // range, where accuracy falls away quickly, as it does for anyone), and
  // mistiming the ball.
  const real over = std::clamp((in.swing_power - 0.82f) / 0.18f, 0.0f, 1.0f);
  const real scatter = 0.015f + 0.11f * over * over + (mistimed ? 0.05f : 0.0f);
  const auto wobble = [&] { return (next_unit(m.rng) + next_unit(m.rng) - 1.0f) * scatter * 2.0f; };
  const Vec3 target{std::clamp(aim * limit, -limit, limit) + wobble(),
                    m.world.table.height + m.world.ball_params.radius, target_z + wobble()};
  m.last_target = target;

  // Travelling toward -z, topspin is rotation about -x.
  // A hard-hit ball also carries more topspin, as it does off a real bat.
  // That is what lets it be fast and still come down on the table. An upward
  // brush adds a lot more; a downward cut replaces it with backspin.
  ball.spin = {-(40.0f + 90.0f * power) * (1.0f - down) - 140.0f * up + 110.0f * down, 0, 0};
  if (const auto v = plan_shot(m.world, ball.pos, ball.spin, target, speed)) {
    ball.vel = *v;
    ball.vel.y += (next_unit(m.rng) + next_unit(m.rng) - 1.0f) * 0.45f * over * over;
  }
}

void assist_swing_serve(Match& m) {
  BallState& ball = m.world.ball;
  const real aim_x = std::clamp(ball.pos.x + 0.5f * m.input.swing_dir.x, -0.5f, 0.5f);
  if (const auto plan = plan_serve(m.world, ball.pos, aim_x, m.input.swing_power)) {
    ball.vel = plan->vel;
    ball.spin = plan->spin;
  }
}

// The older model, for a bat used as a plain physical paddle: the ball has
// already bounced off it, and the result is re-lofted so that it lands.
void assist_player_stroke(Match& m) {
  if (m.input.swing_mode) return assist_swing_stroke(m);
  BallState& ball = m.world.ball;
  if (ball.vel.z > -0.5f) return;  // not going forward at all: leave it as struck

  const Paddle& bat = m.paddles[index(Side::Player)];
  const real struck = std::hypot(ball.vel.x, ball.vel.z);
  const real speed = std::clamp(std::max(struck, 5.5f + 0.9f * length(bat.vel)), 5.5f, 11.0f);
  const real power = (speed - 5.5f) / (11.0f - 5.5f);
  const real half_l = m.world.table.length * 0.5f;
  const real target_z = -(0.40f + 0.48f * power) * half_l;

  const real slope = ball.vel.x / -ball.vel.z;  // sideways travel per metre forward
  const real steer = std::clamp(0.16f * bat.vel.x, -0.45f, 0.45f);
  const real limit = m.world.table.width * 0.5f - 0.12f;
  const Vec3 target{std::clamp(ball.pos.x + slope * (ball.pos.z - target_z) + steer, -limit, limit),
                    m.world.table.height + m.world.ball_params.radius, target_z};

  if (const auto v = plan_shot(m.world, ball.pos, ball.spin, target, speed)) ball.vel = *v;
}

void assist_player_serve(Match& m) {
  if (m.input.swing_mode) return assist_swing_serve(m);
  BallState& ball = m.world.ball;
  const real speed = std::hypot(ball.vel.x, ball.vel.z);
  const real pace = std::clamp((speed - 3.0f) / 6.0f, real{0}, real{1});
  const real slope = ball.vel.z < -0.5f ? ball.vel.x / -ball.vel.z : real{0};
  const real aim_x = std::clamp(ball.pos.x + slope * 2.0f, -0.5f, 0.5f);
  if (const auto plan = plan_serve(m.world, ball.pos, aim_x, pace)) {
    ball.vel = plan->vel;
    ball.spin = plan->spin;
  }
}

// The opponent picks a target and a pace. It is beatable the way a real
// player is:
//  - every stroke lands a little off its target, more so off a fast ball;
//  - now and then it makes an outright error (long, wide, or into the net).
//    Seldom off an easy ball; more often off a fast one, off one with heavy
//    spin, and when it had to lunge to reach it. Spin also decides the kind
//    of error, as it does on a real bat: topspin kicks up off the rubber and
//    the return flies long, backspin drags it down into the net.
void opponent_stroke(Match& m, real incoming_speed) {
  BallState& ball = m.world.ball;
  const Table& t = m.world.table;
  const real half_w = t.width * 0.5f, half_l = t.length * 0.5f;
  const real sigma = 0.04f + 0.010f * incoming_speed;
  auto noise = [&] { return (next_unit(m.rng) + next_unit(m.rng) - 1.0f) * sigma * 2.0f; };

  const OpponentSkill& sk = skill(m);
  Vec3 target{next_range(m.rng, -sk.width, sk.width) * half_w + noise(),
              t.height + m.world.ball_params.radius,
              next_range(m.rng, 0.50f, 0.80f) * half_l + noise()};
  // A thinking opponent plays to where the player is not: if the bat is on
  // one side, the ball goes to the other. (This is what makes moving the
  // hand matter: the bat only goes to the ball from roughly the right side.)
  if (next_unit(m.rng) < sk.placement) {
    const real bat_x = m.paddles[index(Side::Player)].pos.x;
    const real away = std::abs(bat_x) > 0.08f ? -std::copysign(real{1}, bat_x)
                                              : (next_unit(m.rng) < 0.5f ? real{-1} : real{1});
    target.x = away * next_range(m.rng, 0.5f * sk.width, sk.width) * half_w + noise();
  }
  // Brisk and flat, with only light topspin: a slow ball has to be lofted to
  // clear the net and then sits up high off the bounce.
  ball.spin = {next_range(m.rng, 15.0f, 60.0f), 0, 0};  // topspin for travel toward +z
  const real speed = next_range(m.rng, sk.pace_lo, sk.pace_hi);

  // Toward +z the player's topspin arrives as rotation about -x.
  const real topspin = std::max(-ball.spin.x, real{0}), backspin = std::max(ball.spin.x, real{0});
  const real lunging = length(m.paddles[index(Side::Opponent)].vel) / sk.speed;  // 0..1
  const real error_chance =
      // (Rates halved from the first version: with rallies now running long,
      // that one decided more than half of all points by itself.)
      std::clamp(sk.errors * (0.012f + 0.006f * incoming_speed + 0.0004f * (topspin + backspin) +
                              0.10f * lunging),
                 0.02f, 0.40f);
  enum { Clean, Long, Wide, Netted } error = Clean;
  if (next_unit(m.rng) < error_chance) {
    const real into_net = backspin > 30.0f ? 0.70f : topspin > 120.0f ? 0.15f : 0.35f;
    const real kind = next_unit(m.rng);
    error = kind < into_net ? Netted : kind < into_net + 0.2f * (1.0f - into_net) ? Wide : Long;
  }
  if (error == Long) target.z = half_l + next_range(m.rng, 0.05f, 0.40f);
  if (error == Wide) target.x = std::copysign(half_w + next_range(m.rng, 0.04f, 0.30f), target.x);

  // A clean stroke is chosen, not rolled: the planner flies several hundred
  // candidate shots and plays the one that is hardest for this player right
  // now (see game/opponent_planner.hpp). The stroke then comes off the bat a
  // touch different from what was intended, more so off a fast ball, which
  // is how a shot aimed near a line sometimes misses it.
  if (error == Clean) {
    const real bat_x = m.paddles[index(Side::Player)].pos.x;
    if (const auto shot = opponent_planner().choose(m.world, sk, bat_x, m.rng)) {
      const real slip = 0.004f + 0.0015f * incoming_speed;
      ball.vel = {shot->vel.x + noise() * 0.6f, shot->vel.y * (1.0f + slip * (2.0f * next_unit(m.rng) - 1.0f)),
                  shot->vel.z * (1.0f + slip * (2.0f * next_unit(m.rng) - 1.0f))};
      ball.spin = shot->spin;
      return;
    }
  }
  // An error, or (rarely) nothing playable from here: the single-shot solver.
  if (const auto v = plan_shot(m.world, ball.pos, ball.spin, target, speed)) {
    ball.vel = *v;
    if (error == Netted) ball.vel.y -= next_range(m.rng, 0.5f, 1.2f);  // not enough lift
  }
}

void opponent_serve(Match& m) {
  BallState& ball = m.world.ball;
  const real aim_x = next_range(m.rng, -0.4f, 0.4f);
  const real pace = next_range(m.rng, 0.55f, 0.95f);  // flat serves bounce low

  auto plan = plan_serve(m.world, ball.pos, aim_x, pace);
  if (!plan) {  // awkward paddle position: serve from the standard spot instead
    ball.pos = opponent_ready(m) + Vec3{0, 0.05f, 0.08f};
    plan = plan_serve(m.world, ball.pos, aim_x, pace);
  }
  ball.vel = plan ? plan->vel : Vec3{0, 0, 3};
  ball.spin = plan ? plan->spin : Vec3{};
  m.phase = Phase::Rally;
  m.stage = Stage::ServeOwnBounce;
  m.striker = Side::Opponent;
  m.quiet = 0;
}

// ------------------------------------------------------------------ opponent

void move_opponent(Match& m, real dt) {
  Paddle& o = m.paddles[index(Side::Opponent)];
  const bool incoming = m.phase == Phase::Rally && m.striker == Side::Player &&
                        m.stage != Stage::ServeToss;

  if (!incoming) {
    m.ai_goal = opponent_ready(m);
  } else if (m.ai_replan) {
    m.ai_wait -= dt;
    if (m.ai_wait <= 0) {
      m.ai_replan = false;
      const Vec3 plane{0, 0, kOpponentZ + m.world.ball_params.radius};
      const auto c = predict_crossing(m.world, plane, /*toward_ball=*/{0, 0, 1},
                                      /*already_bounced=*/m.stage == Stage::AwaitHit);
      // Only go for balls that will bounce first. A ball arriving on the full
      // is going out, and touching it would only risk giving the point away.
      if (c && c->bounced) {
        m.ai_goal = {c->point.x, std::max(c->point.y, m.world.table.height + 0.03f), kOpponentZ};
      } else if (c) {
        m.ai_goal = {c->point.x + (c->point.x > 0 ? -0.5f : 0.5f), o.pos.y, kOpponentZ};
      }
    }
  }

  const Vec3 delta = m.ai_goal - o.pos;
  const real dist = length(delta);
  const real max_step = skill(m).speed * dt;
  o.vel = (dist > max_step ? delta * (max_step / dist) : delta) / dt;
}

// --------------------------------------------------------------- the rules

void on_paddle(Match& m, Side who, real incoming_speed) {
  if (m.hit_cooldown[index(who)] > 0) return;
  m.hit_cooldown[index(who)] = kHitCooldown;
  m.quiet = 0;

  switch (m.stage) {
    case Stage::ServeToss:
      if (who != m.server) return;
      m.stage = Stage::ServeOwnBounce;
      m.striker = who;
      m.serve_touched_net = false;
      ++m.rally_hits;
      if (who == Side::Player && m.assist) assist_player_serve(m);
      break;

    case Stage::AwaitHit:
      if (who == m.striker) return;  // the ball came back to its own striker; play on
      m.striker = who;
      m.stage = Stage::AwaitBounce;
      ++m.rally_hits;
      if (who == Side::Player) {
        if (m.assist) assist_player_stroke(m);
      } else {
        opponent_stroke(m, incoming_speed);
      }
      break;

    case Stage::ServeOwnBounce:
    case Stage::ServeFarBounce:
    case Stage::AwaitBounce:
      if (who == m.striker) return;  // second touch of one stroke
      // Touched before it bounced on the receiver's half.
      if (m.stage == Stage::ServeOwnBounce || over_table(m, m.world.ball.pos)) {
        award(m, m.striker, PointReason::Volley);
      } else {
        award(m, who, PointReason::Out);  // it had already missed the table
      }
      return;
  }

  if (who == Side::Player) {
    ++m.player_hits;
    m.last_swing_age = m.input.swing_age;
    m.ai_replan = true;
    m.ai_wait = skill(m).reaction;
    m.last_hit_speed = length(m.world.ball.vel);
  }
}

// Swing play: is the player striking the ball on this tick? The swing must be
// live, the ball within reach, and it must be the player's ball to hit (after
// its bounce in a rally, or on its way down from the toss when serving).
bool swing_connects(const Match& m) {
  const PaddleInput& in = m.input;
  if (!in.swing_mode || !in.swinging || m.hit_cooldown[index(Side::Player)] > 0) return false;
  const BallState& ball = m.world.ball;
  const Vec3 gap = ball.pos - in.pos;  // +z is behind the bat, toward the player
  if (gap.x * gap.x + gap.y * gap.y > kSwingReach * kSwingReach) return false;
  if (gap.z < -kSwingReachFront || gap.z > kSwingReachBehind) return false;
  if (m.stage == Stage::ServeToss) return m.server == Side::Player && ball.vel.y < 0;
  return m.stage == Stage::AwaitHit && m.striker == Side::Opponent;
}

void on_table(Match& m, Side half) {
  m.quiet = 0;
  if (m.striker == Side::Player) m.ai_replan = true;  // the bounce changes the path

  switch (m.stage) {
    case Stage::ServeToss:
      replay(m, PointReason::None);
      break;
    case Stage::ServeOwnBounce:
      if (half == m.striker) m.stage = Stage::ServeFarBounce;
      else award(m, other(m.striker), PointReason::ServeFault);
      break;
    case Stage::ServeFarBounce:
      if (half == m.striker) award(m, other(m.striker), PointReason::ServeFault);
      else if (m.serve_touched_net) replay(m, PointReason::Let);
      else m.stage = Stage::AwaitHit;
      break;
    case Stage::AwaitBounce:
      if (half == m.striker) award(m, other(m.striker), PointReason::OwnSide);
      else m.stage = Stage::AwaitHit;
      break;
    case Stage::AwaitHit:
      award(m, m.striker, PointReason::DoubleBounce);
      break;
  }
}

// The ball left play: floor, far out of bounds, or dead on the table.
void on_dead_ball(Match& m) {
  switch (m.stage) {
    case Stage::ServeToss:
      replay(m, PointReason::None);
      break;
    case Stage::ServeOwnBounce:
    case Stage::ServeFarBounce:
    case Stage::AwaitBounce: {
      // The striker never reached the other half. Call it by where it died.
      const bool own_side = (m.world.ball.pos.z > 0) == (m.striker == Side::Player);
      const bool serving = m.stage != Stage::AwaitBounce;
      award(m, other(m.striker),
            serving ? PointReason::ServeFault : own_side ? PointReason::Net : PointReason::Out);
      break;
    }
    case Stage::AwaitHit:
      award(m, m.striker, PointReason::Missed);
      break;
  }
}

}  // namespace

const char* difficulty_name(int level) {
  static constexpr const char* kNames[kDifficultyLevels] = {"Beginner", "Casual", "Club", "Strong",
                                                            "Expert"};
  return kNames[std::clamp(level, 0, kDifficultyLevels - 1)];
}

const char* describe(PointReason r) {
  switch (r) {
    case PointReason::None: return "";
    case PointReason::Let: return "Let: serve touched the net";
    case PointReason::ServeFault: return "Service fault";
    case PointReason::Net: return "Into the net";
    case PointReason::Out: return "Out";
    case PointReason::OwnSide: return "Bounced on own side";
    case PointReason::DoubleBounce: return "Double bounce";
    case PointReason::Missed: return "Not returned";
    case PointReason::Volley: return "Hit before the bounce";
  }
  return "";
}

Side server_for(int points_played, Side first_server) {
  // Two serves each up to 10-10 (20 points), then one each.
  const int turn = points_played < 20 ? (points_played / 2) % 2 : points_played % 2;
  return turn == 0 ? first_server : other(first_server);
}

bool game_won(int winner_score, int loser_score) {
  return winner_score >= 11 && winner_score - loser_score >= 2;
}

Match::Match(Side first, std::uint32_t seed) : first_server(first), server(first), rng(seed | 1u) {
  Paddle& player = paddles[index(Side::Player)];
  player.normal = normalized(Vec3{0, 0.3f, -1});
  player.radius = 0.11f;
  player.pos = {0, world.table.height + 0.25f, 1.55f};

  Paddle& opponent = paddles[index(Side::Opponent)];
  opponent.normal = normalized(Vec3{0, 0.2f, 1});
  opponent.radius = 0.10f;
  opponent.pos = opponent_ready(*this);
  ai_goal = opponent.pos;

  begin_serve(*this);
}

// The ball starts on the paddle, as it would on an open palm, and is thrown
// straight up. If the paddle stays where it is, the ball comes back down onto
// it: holding still is a serve.
Vec3 toss_point(const Match& m, Vec3 paddle_pos) {
  const real behind_end_line = m.world.table.length * 0.5f + 0.04f;  // the law: not over the table
  return {paddle_pos.x, paddle_pos.y + 0.02f, std::max(paddle_pos.z - 0.035f, behind_end_line)};
}

ContactMask tick(Match& m, const PaddleInput& input, real dt) {
  Paddle& player = m.paddles[index(Side::Player)];
  Paddle& opponent = m.paddles[index(Side::Opponent)];
  m.input = input;
  player.pos = input.pos;
  player.vel = input.vel;
  player.normal = input.normal;
  move_opponent(m, dt);
  // One stroke is one contact. For a moment after a hit the paddle is
  // switched off, so a tilted face moving through the ball's path cannot
  // clip it a second time and spoil the shot.
  for (std::size_t i = 0; i < 2; ++i) {
    m.hit_cooldown[i] = std::max(real{0}, m.hit_cooldown[i] - dt);
    m.paddles[i].enabled = m.hit_cooldown[i] <= 0;
  }
  // In swing play the bat is not a wall: a ball that meets a bat that is not
  // swinging goes straight past it.
  if (input.swing_mode) player.enabled = false;

  ContactMask hit = contact::kNone;
  switch (m.phase) {
    case Phase::PreServe: {
      BallState& ball = m.world.ball;
      m.timer -= dt;
      if (m.server == Side::Player) {
        ball.pos = toss_point(m, player.pos);
        if (m.timer <= 0) {
          ball.vel = {0, kTossSpeed, 0};
          m.phase = Phase::Rally;
          m.quiet = 0;
        }
      } else {
        ball.pos = opponent.pos + Vec3{0, 0.05f, 0.08f};
        if (m.timer <= 0) opponent_serve(m);
      }
      break;
    }

    case Phase::Rally: {
      const real incoming_speed = length(m.world.ball.vel);
      bool strike_now = false;
      if (m.strike_hold > 0) {
        // The ball is on the bat, waiting for the swing to show its strength.
        m.strike_hold -= dt;
        if (m.strike_hold > 0) break;  // nothing else moves the ball meanwhile
        strike_now = true;
      } else if (swing_connects(m)) {
        m.strike_hold = kStrikeHold;
        m.strike_incoming_speed = incoming_speed;
        m.strike_swing_age = m.input.swing_age;
        m.strike_gap_z = m.world.ball.pos.z - m.input.pos.z;
        hit |= contact::kPaddle0;  // the sound of contact belongs to this moment
        break;
      }
      if (strike_now) {
        if (!m.assist) {
          // No help wanted: a plain bounce off the moving bat.
          BallState& ball = m.world.ball;
          const Vec3 toward_ball = normalized(ball.pos - player.pos);
          ball.vel -= player.vel;
          bounce(ball, m.world.ball_params, toward_ball, player.surface);
          ball.vel += player.vel;
        }
        on_paddle(m, Side::Player, m.strike_incoming_speed);
      }
      hit |= step(m.world, m.paddles, dt);
      m.quiet += dt;

      // Same order the physics resolved them in: paddles, net, table, floor.
      if (hit & contact::kPaddle0) on_paddle(m, Side::Player, incoming_speed);
      if (m.phase == Phase::Rally && (hit & contact::kPaddle1)) {
        on_paddle(m, Side::Opponent, incoming_speed);
      }
      if (m.phase == Phase::Rally && (hit & contact::kNet)) {
        m.quiet = 0;
        if (m.stage == Stage::ServeOwnBounce || m.stage == Stage::ServeFarBounce) {
          m.serve_touched_net = true;
        }
      }
      if (m.phase == Phase::Rally && (hit & contact::kTableNear)) on_table(m, Side::Player);
      if (m.phase == Phase::Rally && (hit & contact::kTableFar)) on_table(m, Side::Opponent);

      const Vec3 p = m.world.ball.pos;
      const bool dead = (hit & contact::kFloor) || m.quiet > kQuietLimit ||
                        std::abs(p.x) > 4.0f || std::abs(p.z) > 5.0f;
      if (m.phase == Phase::Rally && dead) on_dead_ball(m);
      break;
    }

    case Phase::PointOver:
      hit |= step(m.world, {}, dt);  // let the dead ball bounce away (and be heard)
      m.timer -= dt;
      if (m.timer <= 0) {
        if (m.last_reason == PointReason::None || m.last_reason == PointReason::Let) {
          begin_serve(m);  // replay: same server, no point scored
        } else {
          after_point(m);
        }
      }
      break;

    case Phase::GameOver:
      hit |= step(m.world, {}, dt);
      break;
  }

  opponent.pos += opponent.vel * dt;
  return hit;
}

}  // namespace pp
