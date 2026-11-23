#include "game/match.hpp"

#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>

using namespace pp;

namespace {

constexpr real kDt = 1e-3f;

// A paddle parked far away, for when the player should do nothing.
const PaddleInput kAbsent{.pos = {0, 5, 5}, .vel = {}, .normal = {0, 0, -1}};

// Runs until the current point is decided (or replayed).
void play_out(Match& m, const PaddleInput& input, int max_ticks = 8000) {
  for (int i = 0; i < max_ticks && m.phase != Phase::PointOver; ++i) tick(m, input, kDt);
}

// Puts a match straight into a rally with the ball where the test wants it.
Match rally(Side striker, Stage stage, BallState ball) {
  Match m;
  m.phase = Phase::Rally;
  m.stage = stage;
  m.striker = striker;
  m.world.ball = ball;
  return m;
}

// A competent scripted player: reads where the ball will arrive, puts the
// paddle there and pushes through it. Serves by holding still under the toss.
struct Bot {
  Vec3 pos{0, 1.0f, 1.55f};
  int since_plan = 0;

  PaddleInput play(const Match& m) {
    Vec3 want{0, 1.0f, 1.55f};
    Vec3 vel{};

    if (m.phase == Phase::Rally && m.stage == Stage::ServeToss && m.server == Side::Player) {
      want = pos;  // the toss comes back down onto a paddle that holds still
    } else if (m.phase == Phase::Rally && m.striker == Side::Opponent) {
      if (since_plan == 0) ++incoming;  // a new ball is on its way
      const bool skip = skip_every > 0 && incoming % skip_every == 0;
      if (skip) {
        since_plan = 1;
        has_target = false;
      } else if (++since_plan >= 25 || since_plan == 1) {
        since_plan = 1;
        // Read the arrival point just behind the paddle, so the target stays
        // valid right up to contact.
        if (const auto c = predict_crossing(m.world, {0, 0, 1.58f}, {0, 0, -1},
                                            m.stage == Stage::AwaitHit);
            c && c->bounced) {
          target = c->point;
          has_target = true;
        } else {
          has_target = false;
        }
      }
      if (has_target) {
        want = {target.x, target.y, 1.55f};
        vel = {0, 0, -2.5f};
      }
    } else {
      since_plan = 0;
      has_target = false;
      want = pos;  // stay put between strokes; a real hand does not teleport
    }
    pos = want;
    return {.pos = pos, .vel = vel, .normal = normalized(Vec3{0, 0.3f, -1})};
  }

  Vec3 target;
  bool has_target = false;
  int skip_every = 0;  // 3 = lets every third ball go by; 0 = never misses
  int incoming = 0;
};

}  // namespace

// ------------------------------------------------------------------ scoring

TEST_CASE("serve changes every two points, then every point from 10-10", "[rules]") {
  const Side a = Side::Player, b = Side::Opponent;
  CHECK(server_for(0, a) == a);
  CHECK(server_for(1, a) == a);
  CHECK(server_for(2, a) == b);
  CHECK(server_for(3, a) == b);
  CHECK(server_for(4, a) == a);
  CHECK(server_for(19, a) == b);
  CHECK(server_for(20, a) == a);  // 10-10
  CHECK(server_for(21, a) == b);
  CHECK(server_for(22, a) == a);
  CHECK(server_for(0, b) == b);
}

TEST_CASE("a game is won at 11 with a two-point lead", "[rules]") {
  CHECK(game_won(11, 9));
  CHECK(game_won(11, 0));
  CHECK_FALSE(game_won(10, 3));
  CHECK_FALSE(game_won(11, 10));
  CHECK(game_won(12, 10));
  CHECK_FALSE(game_won(15, 14));
  CHECK(game_won(16, 14));
}

// -------------------------------------------------------------------- serve

TEST_CASE("the opponent's serve is always legal; unreturned, it wins the point", "[rules][serve]") {
  for (std::uint32_t seed = 1; seed <= 150; ++seed) {
    Match m{Side::Opponent, seed * 2654435761u};
    play_out(m, kAbsent);
    INFO("seed " << seed << ": " << describe(m.last_reason));
    REQUIRE(m.phase == Phase::PointOver);
    // A let is legal too (replayed). Otherwise the serve landed, and then
    // either bounced again or ran off the end with nobody there to return it.
    if (m.last_reason != PointReason::Let) {
      const bool ace = m.last_reason == PointReason::DoubleBounce ||
                       m.last_reason == PointReason::Missed;
      REQUIRE(ace);
      REQUIRE(m.score[index(Side::Opponent)] == 1);
    }
  }
}

TEST_CASE("holding the paddle still under the toss is a legal serve", "[rules][serve]") {
  // The paddle height decides where the ball is struck; all of these must work.
  const real height = GENERATE(0.10f, 0.20f, 0.32f, 0.45f);
  const real x = GENERATE(-0.5f, 0.0f, 0.5f);
  Match m{Side::Player};
  const PaddleInput hold{.pos = {x, m.world.table.height + height, 1.50f},
                         .vel = {},
                         .normal = normalized(Vec3{0, 0.3f, -1})};
  for (int i = 0; i < 4000 && !(m.phase == Phase::Rally && m.stage == Stage::AwaitHit); ++i) {
    tick(m, hold, kDt);
    REQUIRE(m.phase != Phase::PointOver);  // no fault, no missed toss
  }
  CHECK(m.stage == Stage::AwaitHit);  // bounced on our half, then on theirs
  CHECK(m.striker == Side::Player);
}

TEST_CASE("a toss nobody hits is tossed again without a point", "[rules][serve]") {
  Match m{Side::Player};
  const PaddleInput ready{.pos = {0, 1.0f, 1.55f}, .vel = {}, .normal = normalized(Vec3{0, 0.3f, -1})};
  while (m.phase == Phase::PreServe) tick(m, ready, kDt);
  REQUIRE(m.stage == Stage::ServeToss);
  play_out(m, kAbsent);  // the player pulls the paddle away after the toss
  REQUIRE(m.phase == Phase::PointOver);
  for (int i = 0; i < 1000; ++i) tick(m, kAbsent, kDt);
  CHECK(m.score[0] + m.score[1] == 0);
  CHECK(m.server == Side::Player);
}

TEST_CASE("a serve that goes straight over without bouncing first is a fault", "[rules][serve]") {
  const Match fresh;
  const real y = fresh.world.table.height + 0.3f;
  Match m = rally(Side::Player, Stage::ServeOwnBounce,
                  {.pos = {0, y, 1.3f}, .vel = {0, 2.0f, -4.5f}, .spin = {}});
  play_out(m, kAbsent);
  CHECK(m.last_reason == PointReason::ServeFault);
  CHECK(m.last_winner == Side::Opponent);
}

// -------------------------------------------------------------------- rally

TEST_CASE("the ball bouncing twice on your side loses the point", "[rules]") {
  Match m = rally(Side::Opponent, Stage::AwaitBounce,
                  {.pos = {0, 1.0f, 0.3f}, .vel = {0, 0, 1.0f}, .spin = {}});
  play_out(m, kAbsent);
  CHECK(m.last_reason == PointReason::DoubleBounce);
  CHECK(m.last_winner == Side::Opponent);
}

TEST_CASE("one bounce and then off the end, untouched, also loses the point", "[rules]") {
  Match m = rally(Side::Opponent, Stage::AwaitBounce,
                  {.pos = {0, 1.0f, 0.3f}, .vel = {0, 0, 5.0f}, .spin = {}});
  play_out(m, kAbsent);
  CHECK(m.last_reason == PointReason::Missed);
  CHECK(m.last_winner == Side::Opponent);
}

TEST_CASE("hitting your own half first loses the point", "[rules]") {
  Match m = rally(Side::Player, Stage::AwaitBounce,
                  {.pos = {0, 1.0f, 1.0f}, .vel = {0, 0, -1.0f}, .spin = {}});
  play_out(m, kAbsent);
  CHECK(m.last_reason == PointReason::OwnSide);
  CHECK(m.last_winner == Side::Opponent);
}

TEST_CASE("a stroke that flies off the end of the table loses the point", "[rules]") {
  Match m = rally(Side::Player, Stage::AwaitBounce,
                  {.pos = {0, 1.1f, 1.4f}, .vel = {0, 3.0f, -9.0f}, .spin = {}});
  play_out(m, kAbsent);
  CHECK(m.last_reason == PointReason::Out);
  CHECK(m.last_winner == Side::Opponent);
}

TEST_CASE("a ball that dies in the net is called as such", "[rules]") {
  // Launched from over the player's half, too low: hits the net, drops on the player's side.
  Match m = rally(Side::Player, Stage::AwaitBounce, {});
  m.world.ball = {.pos = {0, m.world.table.height + 0.10f, 0.5f}, .vel = {0, 0.5f, -5.0f}, .spin = {}};
  play_out(m, kAbsent);
  CHECK(m.last_winner == Side::Opponent);
  CHECK((m.last_reason == PointReason::OwnSide || m.last_reason == PointReason::Net));
}

TEST_CASE("volleying over the table loses the point", "[rules]") {
  // Ball flying over the player's half; the paddle reaches in over the table.
  Match m = rally(Side::Opponent, Stage::AwaitBounce,
                  {.pos = {0, 1.1f, 0.2f}, .vel = {0, 0.5f, 5.0f}, .spin = {}});
  const PaddleInput reach_in{.pos = {0, 1.15f, 1.0f}, .vel = {}, .normal = {0, 0, -1}};
  play_out(m, reach_in);
  CHECK(m.last_reason == PointReason::Volley);
  CHECK(m.last_winner == Side::Opponent);
}

TEST_CASE("catching a ball that already missed the table wins the point", "[rules]") {
  // Same ball, but the player waits behind the end line: it was going long.
  Match m = rally(Side::Opponent, Stage::AwaitBounce,
                  {.pos = {0, 1.1f, 0.2f}, .vel = {0, 1.5f, 7.0f}, .spin = {}});
  const auto c = predict_crossing(m.world, {0, 0, 1.6f}, {0, 0, -1});
  REQUIRE(c.has_value());
  REQUIRE_FALSE(c->bounced);
  const PaddleInput behind{.pos = {c->point.x, c->point.y, 1.62f}, .vel = {}, .normal = {0, 0, -1}};
  play_out(m, behind);
  CHECK(m.last_reason == PointReason::Out);
  CHECK(m.last_winner == Side::Player);
}

// ---------------------------------------------------------------- whole game

namespace {

struct Totals {
  int player = 0, opponent = 0, games = 0, long_rallies = 0;
};

// Plays whole games and checks each one ends with a legal final score.
Totals play_games(int count, int skip_every) {
  Totals totals;
  for (int g = 1; g <= count; ++g) {
    const auto seed = static_cast<std::uint32_t>(g);
    Match m{g % 2 ? Side::Player : Side::Opponent, seed * 40503u};
    Bot bot;
    bot.skip_every = skip_every;
    int max_rally = 0;
    for (int i = 0; i < 900000 && m.phase != Phase::GameOver; ++i) {
      tick(m, bot.play(m), kDt);
      max_rally = std::max(max_rally, m.rally_hits);
    }
    INFO("game " << g << " score " << m.score[0] << "-" << m.score[1]);
    REQUIRE(m.phase == Phase::GameOver);
    const int hi = std::max(m.score[0], m.score[1]), lo = std::min(m.score[0], m.score[1]);
    CHECK(game_won(hi, lo));
    CHECK((hi == 11 || hi - lo == 2));  // nobody scores after the game is won
    totals.player += m.score[0];
    totals.opponent += m.score[1];
    totals.long_rallies += max_rally >= 6;
    ++totals.games;
  }
  return totals;
}

}  // namespace

TEST_CASE("a player who reaches every ball wins, through real rallies", "[match]") {
  // With the assist on, getting the paddle to the ball is the whole job. A
  // bot that always does must never lose a point to its own stroke; it wins
  // when the opponent errs.
  const Totals t = play_games(4, /*skip_every=*/0);
  INFO("player " << t.player << ", opponent " << t.opponent);
  CHECK(t.long_rallies == t.games);
  CHECK(t.player >= 11 * t.games);
  CHECK(t.opponent <= t.games);  // at most the odd freak point
}

TEST_CASE("a player who lets balls go by loses points, and can lose the game", "[match]") {
  const Totals t = play_games(4, /*skip_every=*/3);
  INFO("player " << t.player << ", opponent " << t.opponent);
  CHECK(t.opponent >= 20);
  // It still takes some points: the opponent does err, though seldom now
  // (its unforced error rate was halved once rallies grew long).
  CHECK(t.player >= 4);
}
