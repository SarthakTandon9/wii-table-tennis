#include "game/swing.hpp"

#include "game/match.hpp"
#include "game/meeting_point.hpp"
#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>

#include <optional>

using namespace pp;

namespace {

constexpr real kDt = 1e-3f;

// Feeds the detector a constant speed for a while; returns how many swings began.
int run(SwingDetector& d, real speed, real seconds) {
  int started = 0;
  for (real t = 0; t < seconds; t += kDt) started += d.update({speed, 0, 0}, kDt);
  return started;
}

}  // namespace

// ----------------------------------------------------------------- detector

TEST_CASE("ordinary movement is not a swing; a burst of speed is", "[swing]") {
  SwingDetector d{kHandSwing};
  CHECK(run(d, 0.6f, 1.0f) == 0);  // positioning the bat, however long
  CHECK_FALSE(d.live());

  CHECK(run(d, 1.5f, 0.05f) == 1);
  CHECK(d.live());
}

TEST_CASE("a swing is live for a limited time, then needs the hand to slow before the next", "[swing]") {
  SwingDetector d{kHandSwing};
  // Keep moving fast for two seconds: that is one swing, not five.
  CHECK(run(d, 2.0f, 2.0f) == 1);
  CHECK_FALSE(d.live());  // long expired

  CHECK(run(d, 0.2f, 0.3f) == 0);  // slow down: re-armed, recovered
  CHECK(run(d, 2.0f, 0.05f) == 1);
}

TEST_CASE("a swing that connects is spent at once and the next may follow", "[swing]") {
  SwingDetector d{kHandSwing};
  REQUIRE(run(d, 2.0f, 0.05f) == 1);
  d.consume();
  CHECK_FALSE(d.live());
  CHECK(run(d, 0.2f, 0.02f) == 0);   // a brief slow-down between strokes
  CHECK(run(d, 2.0f, 0.02f) == 1);   // no recovery delay after a hit
}

TEST_CASE("power follows the fastest moment of the swing, direction too", "[swing]") {
  SwingDetector gentle{kHandSwing}, full{kHandSwing};
  run(gentle, kHandSwing.start_speed, 0.05f);
  CHECK(gentle.power() == 0.0f);

  // Starts slow, peaks later, going up and to the left.
  full.update({-0.7f, 0.7f, 0}, kDt);
  for (int i = 0; i < 50; ++i) full.update({-3.5f, 3.5f, 0}, kDt);
  CHECK(full.power() == 1.0f);
  CHECK(full.direction().x < -0.6f);
  CHECK(full.direction().y > 0.6f);
}

// -------------------------------------------------------- swinging in a match

namespace {

struct SwingPlan {
  std::optional<real> lead;  // begin swinging this long before the ball arrives; none = never
  real power = 0.5f;
  Vec3 dir{0, 0, 0};
  real face_turn = 0;
  std::uint32_t seed = 0xABCDu;
};

struct Outcome {
  int player_hits = 0;
  bool reached_opponent = false;  // the return bounced on their half
  Vec3 landing;
  real speed_after_hit = 0;
  real spin_after_hit = 0;  // about x: negative is topspin for a ball going away
  StrokeKind stroke = StrokeKind::Drive;
  StrokeTiming timing = StrokeTiming::Good;
  Vec3 target;
  PointReason reason = PointReason::None;
  Side winner = Side::Player;
};

// Receives the opponent's serve with the bat on the meeting point, swinging
// (or not) as planned. Plays until the point is decided or the return lands.
Outcome receive_serve(const SwingPlan& plan) {
  Match m{Side::Opponent, plan.seed};
  const real plane_z = 1.50f, live_for = kHandSwing.live_for;
  Vec3 bat{0, 0.95f, plane_z};
  real swing_age = -1;  // < 0: not swinging
  Outcome out;

  for (int i = 0; i < 12000; ++i) {
    real due = 100;
    if (m.phase == Phase::Rally && m.striker == Side::Opponent) {
      if (const auto c = predict_crossing(m.world, {0, 0, plane_z}, {0, 0, -1},
                                          m.stage == Stage::AwaitHit)) {
        if (c->bounced) bat = c->point;
        due = c->time;
      } else if (m.world.ball.pos.z > plane_z) {
        due = -(m.world.ball.pos.z - plane_z) / std::max(m.world.ball.vel.z, 0.1f);  // already past
      }
    }
    if (swing_age < 0 && plan.lead && due <= *plan.lead) swing_age = 0;
    const bool live = swing_age >= 0 && swing_age < live_for;

    const int hits_before = m.player_hits;
    tick(m, {.pos = bat, .vel = {}, .normal = {0, 0, -1}, .swing_mode = true, .swinging = live,
             .swing_power = plan.power, .swing_dir = plan.dir,
             .swing_age = std::max(swing_age, real{0}), .face_turn = plan.face_turn},
         kDt);
    if (swing_age >= 0) swing_age += kDt;

    if (m.player_hits != hits_before) {
      out.speed_after_hit = length(m.world.ball.vel);
      out.spin_after_hit = m.world.ball.spin.x;
      out.stroke = m.last_stroke;
      out.timing = m.last_timing;
      out.target = m.last_target;
    }
    out.player_hits = m.player_hits;
    if (m.striker == Side::Player && m.stage == Stage::AwaitHit) {
      out.reached_opponent = true;
      out.landing = m.world.ball.pos;
      return out;
    }
    if (m.phase == Phase::PointOver) {
      out.reason = m.last_reason;
      out.winner = m.last_winner;
      return out;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("having the bat on the ball is not a hit: no swing, no return", "[swing][match]") {
  const Outcome o = receive_serve({.lead = std::nullopt});
  CHECK(o.player_hits == 0);
  CHECK(o.winner == Side::Opponent);
  CHECK(o.reason == PointReason::Missed);
}

TEST_CASE("a swing shortly before the ball arrives returns it", "[swing][match]") {
  // Anywhere in the live window works: just before arrival, or well ahead.
  const real lead = GENERATE(0.02f, 0.10f, 0.20f, 0.35f);
  const Outcome o = receive_serve({.lead = lead});
  INFO("swing began " << lead << " s before arrival");
  CHECK(o.player_hits == 1);
  CHECK(o.reached_opponent);
}

TEST_CASE("a swing that is detected a little late still connects", "[swing][match]") {
  // The reach extends slightly behind the bat, to absorb camera delay.
  const Outcome o = receive_serve({.lead = -0.03f});
  CHECK(o.player_hits == 1);
}

TEST_CASE("swinging far too early, or after the ball has gone, is a miss", "[swing][match]") {
  const Outcome early = receive_serve({.lead = 0.70f});  // long over by the time it arrives
  CHECK(early.player_hits == 0);
  CHECK(early.winner == Side::Opponent);

  const Outcome late = receive_serve({.lead = -0.20f});
  CHECK(late.player_hits == 0);
  CHECK(late.winner == Side::Opponent);
}

TEST_CASE("a harder swing sends the ball faster and deeper", "[swing][match]") {
  const Outcome gentle = receive_serve({.lead = 0.10f, .power = 0.0f});
  const Outcome full = receive_serve({.lead = 0.10f, .power = 1.0f});
  REQUIRE((gentle.reached_opponent && full.reached_opponent));
  CHECK(full.speed_after_hit > gentle.speed_after_hit + 2.0f);
  CHECK(full.landing.z < gentle.landing.z - 0.2f);  // further from the net (more negative z)
}

TEST_CASE("swinging sideways steers the ball", "[swing][match]") {
  const Outcome left = receive_serve({.lead = 0.10f, .dir = {-1, 0, 0}});
  const Outcome right = receive_serve({.lead = 0.10f, .dir = {1, 0, 0}});
  REQUIRE((left.reached_opponent && right.reached_opponent));
  CHECK(left.landing.x < -0.15f);
  CHECK(right.landing.x > 0.15f);
}

TEST_CASE("a hit need not be a forward movement: up is topspin, down is a chop", "[swing][match]") {
  const Outcome push = receive_serve({.lead = 0.10f, .power = 0.6f, .dir = {0, 0, 1}});
  const Outcome up = receive_serve({.lead = 0.10f, .power = 0.6f, .dir = {0.2f, 0.98f, 0}});
  const Outcome down = receive_serve({.lead = 0.10f, .power = 0.6f, .dir = {0, -1, 0}});
  // All three are hits, and all three land.
  REQUIRE((push.reached_opponent && up.reached_opponent && down.reached_opponent));
  CHECK(push.stroke == StrokeKind::Drive);
  CHECK(up.stroke == StrokeKind::Topspin);
  CHECK(down.stroke == StrokeKind::Chop);
  CHECK(up.spin_after_hit < push.spin_after_hit - 80.0f);  // much more topspin
  CHECK(down.spin_after_hit > 50.0f);                      // backspin
  CHECK(down.speed_after_hit < push.speed_after_hit - 1.0f);

  // At full power too, including a chop (which must not sail long).
  for (const Vec3 dir : {Vec3{0, 1, 0}, Vec3{0, -1, 0}, Vec3{0.7f, 0.7f, 0}, Vec3{-0.7f, -0.7f, 0}}) {
    const Outcome o = receive_serve({.lead = 0.10f, .power = 1.0f, .dir = dir});
    INFO("direction " << dir.x << ", " << dir.y);
    CHECK(o.reached_opponent);
  }
}

TEST_CASE("the ball stays strikable for a moment after passing, however the clock falls",
          "[swing][meeting]") {
  // The tracker re-runs its forecast now and then as a safety net. Once the
  // ball has gone by, that forecast finds nothing, and acting on it used to
  // drop the ball in the middle of the late-swing window.
  Match m{Side::Opponent, 0xABCDu};
  MeetingPointTracker tracker;
  const PaddleInput idle{.pos = {0.9f, 1.3f, 1.50f}, .vel = {}, .normal = {0, 0, -1}, .swing_mode = true};
  real late_for = 0;
  bool reached = false;
  for (int i = 0; i < 12000 && m.phase != Phase::PointOver; ++i) {
    tick(m, idle, kDt);
    const MeetingPoint meet = tracker.update(m, {0, 0, 1.50f}, {0, 0, -1}, kDt);
    if (meet.hittable && meet.time < 0) {
      reached = true;
      late_for = -meet.time;
      // It is reported at the ball, reaching back a limited way.
      CHECK(std::abs(meet.point.x - m.world.ball.pos.x) < 1e-4f);
      CHECK(meet.point.z <= 1.50f + 0.41f);
    } else if (reached) {
      break;
    }
  }
  REQUIRE(reached);
  CHECK(late_for > 0.20f);
}

TEST_CASE("the ball goes where the swing went, and nowhere else", "[swing][match][aim]") {
  // The same swing, whatever the bat's face is doing and whatever the dice
  // say, sends the ball to the same place within a hand's width.
  real leftmost = 10, rightmost = -10;
  for (const real face : {-0.4f, 0.0f, 0.4f}) {
    for (const std::uint32_t seed : {0xABCDu, 0x1234u, 0x9999u}) {
      const Outcome o = receive_serve(
          {.lead = 0.10f, .power = 0.5f, .dir = {0.7f, 0.2f, 0}, .face_turn = face, .seed = seed});
      REQUIRE(o.player_hits == 1);
      leftmost = std::min(leftmost, o.target.x);
      rightmost = std::max(rightmost, o.target.x);
    }
  }
  CHECK(leftmost > 0.25f);               // a swing to the right goes right
  CHECK(rightmost - leftmost < 0.08f);   // every time

  // More sideways, further across; straight (a push, or straight up), down the middle.
  const Outcome wide = receive_serve({.lead = 0.10f, .power = 0.5f, .dir = {1, 0, 0}});
  const Outcome push = receive_serve({.lead = 0.10f, .power = 0.5f, .dir = {0, 0, 1}});
  const Outcome up = receive_serve({.lead = 0.10f, .power = 0.5f, .dir = {0, 1, 0}});
  CHECK(wide.target.x > rightmost);
  CHECK(std::abs(push.target.x) < 0.05f);
  CHECK(std::abs(up.target.x) < 0.05f);
}

TEST_CASE("timing is told apart: good, early, late", "[swing][match][aim]") {
  const Outcome good = receive_serve({.lead = 0.10f, .power = 0.6f});
  const Outcome early = receive_serve({.lead = 0.36f, .power = 0.6f});
  const Outcome late = receive_serve({.lead = -0.06f, .power = 0.6f});
  REQUIRE((good.player_hits == 1 && early.player_hits == 1 && late.player_hits == 1));
  CHECK(good.timing == StrokeTiming::Good);
  CHECK(early.timing == StrokeTiming::Early);
  CHECK(late.timing == StrokeTiming::Late);
  // A mistimed ball still goes back, with less on it.
  CHECK(early.speed_after_hit < good.speed_after_hit - 0.5f);
  CHECK(late.speed_after_hit < good.speed_after_hit - 0.5f);
}

TEST_CASE("serving: swing as the toss comes down", "[swing][match][serve]") {
  const PaddleInput hold{.pos = {0, 1.0f, 1.50f}, .vel = {}, .normal = {0, 0, -1}, .swing_mode = true};

  SECTION("no swing: the toss is repeated, nobody scores") {
    Match m{Side::Player};
    for (int i = 0; i < 6000; ++i) tick(m, hold, kDt);
    CHECK(m.player_hits == 0);
    CHECK(m.score[0] + m.score[1] == 0);
  }
  SECTION("a swing while the ball is rising does nothing; one as it falls serves") {
    Match m{Side::Player};
    bool served = false;
    for (int i = 0; i < 6000 && !served; ++i) {
      PaddleInput in = hold;
      in.swinging = m.phase == Phase::Rally && m.stage == Stage::ServeToss;  // swing throughout
      in.swing_power = 0.5f;
      const bool rising = m.world.ball.vel.y > 0;
      const int before = m.player_hits;
      tick(m, in, kDt);
      if (m.player_hits != before) {
        CHECK_FALSE(rising);  // the law: the ball must be falling when struck
        served = true;
      }
    }
    REQUIRE(served);
    // And the serve is good: it reaches the receiver's half after one bounce on ours.
    for (int i = 0; i < 3000 && m.stage != Stage::AwaitHit && m.phase == Phase::Rally; ++i) {
      tick(m, hold, kDt);
    }
    CHECK(m.stage == Stage::AwaitHit);
    CHECK(m.striker == Side::Player);
  }
}

// --------------------------------------------------------- meeting-point cache

#include "game/meeting_point.hpp"

TEST_CASE("predicting only on deflection gives the same meeting point as predicting always",
          "[meeting]") {
  // The cheap policy must be indistinguishable from re-running the forward
  // simulation on every single wake-up, for whole rallies.
  Match m{Side::Opponent, 0x77u};
  MeetingPointTracker cached;
  const Vec3 plane{0, 0, 1.50f}, toward{0, 0, -1};
  int compared = 0, hittable = 0;
  real worst_point = 0, worst_time = 0;
  Vec3 bat{0, 0.95f, 1.50f};

  for (int i = 0; i < 60000; ++i) {
    if (m.phase == Phase::GameOver) m = Match{Side::Opponent, 0x77u + static_cast<unsigned>(i)};
    const MeetingPoint got = cached.update(m, plane, toward, kDt);

    const bool incoming = m.phase == Phase::Rally && m.striker == Side::Opponent &&
                          m.stage != Stage::ServeToss;
    const auto want = incoming ? predict_crossing(m.world, plane, toward, m.stage == Stage::AwaitHit)
                               : std::nullopt;
    if (got.known != want.has_value()) {
      // The two may disagree only in the instant the ball passes the bat,
      // where one says "arriving in a millisecond" and the other "gone".
      const real remaining = got.known ? got.time : want->time;
      REQUIRE(remaining < 0.006f);
      continue;
    }
    if (want) {
      REQUIRE(got.hittable == want->bounced);
      worst_point = std::max(worst_point, length(got.point - want->point));
      worst_time = std::max(worst_time, std::abs(got.time - want->time));
      ++compared;
      hittable += got.hittable;
      if (got.hittable) bat = got.point;
    }
    // Keep rallies going: swing at every ball.
    tick(m, {.pos = bat, .vel = {}, .normal = {0, 0, -1}, .swing_mode = true, .swinging = true,
             .swing_power = 0.4f},
         kDt);
  }

  INFO("compared " << compared << " wake-ups; worst disagreement " << worst_point * 1000
                   << " mm, " << worst_time * 1000 << " ms; forward simulations run: "
                   << cached.predictions());
  CHECK(compared > 5000);
  CHECK(hittable > 2000);
  CHECK(worst_point < 0.01f);  // within a centimetre
  CHECK(worst_time < 0.004f);  // within 4 ms
  CHECK(cached.predictions() * 20 < compared);  // and at a small fraction of the cost
}
