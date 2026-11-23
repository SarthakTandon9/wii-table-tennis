#include "game/player_paddle.hpp"

#include "game/match.hpp"
#include "game/paddle_surface.hpp"
#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

using namespace pp;

namespace {

constexpr real kDt = 1e-3f;

struct Rng {
  std::uint32_t state = 0xC0FFEEu;
  real unit() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<real>(state >> 8) / 16777216.0f;
  }
};

// A player who knows roughly where the ball is going but cannot place the
// paddle precisely: each ball, the paddle ends up somewhere within `sloppiness`
// metres of the right spot, and the hand trembles a little on top of that.
// Returns the share of returnable balls actually returned.
double return_rate(bool magnet_on, real sloppiness, int balls) {
  Match m{Side::Opponent, 0xBADC0DEu};
  PlayerPaddle paddle;
  paddle.magnet.enabled = magnet_on;
  Rng rng;

  const real plane_z = 1.50f;
  Vec3 rest{0, 1.0f, plane_z};
  Vec3 error{};
  int chances = 0, returns = 0, since_predict = 0;
  bool was_incoming = false;
  Stage last_stage = m.stage;
  Side last_striker = m.striker;

  for (int tick_no = 0; tick_no < 4'000'000 && chances < balls; ++tick_no) {
    if (m.phase == Phase::GameOver) m = Match{Side::Opponent, rng.state};

    const bool incoming = m.phase == Phase::Rally && m.striker == Side::Opponent;
    if (incoming && !was_incoming) {  // a new ball: a new aiming error
      const real angle = rng.unit() * 6.2831853f;
      const real radius = sloppiness * std::sqrt(rng.unit());  // uniform over a disc
      error = {radius * std::cos(angle), radius * std::sin(angle), 0};
    }
    was_incoming = incoming;

    if (++since_predict >= 4) {  // as the simulation thread does it
      since_predict = 0;
      if (incoming) {
        const auto c = predict_crossing(m.world, {0, 0, plane_z}, {0, 0, -1},
                                        m.stage == Stage::AwaitHit);
        if (c && c->bounced) {
          const Vec3 tremble{(rng.unit() - 0.5f) * 0.02f, (rng.unit() - 0.5f) * 0.02f, 0};
          paddle.place(c->point + error + tremble, {});
          paddle.set_meeting_point(c->point, c->time);
        } else {
          paddle.clear_meeting_point();
        }
      } else if (m.phase == Phase::Rally && m.stage == Stage::ServeToss) {
        // Our serve: hold still, as the on-screen hint says.
      } else {
        paddle.clear_meeting_point();
        if (m.phase == Phase::PreServe) paddle.place(rest, {});
      }
    }

    const Vec3 at = paddle.pos();
    tick(m, {.pos = at, .vel = paddle.vel(), .normal = synthetic_face_normal(at)}, kDt);
    paddle.advance(kDt, kDt);

    // A chance: the ball has bounced on our half and is ours to hit.
    if (m.phase == Phase::Rally && m.stage == Stage::AwaitHit && m.striker == Side::Opponent &&
        last_stage != Stage::AwaitHit) {
      ++chances;
    }
    if (m.striker == Side::Player && last_striker == Side::Opponent && m.stage == Stage::AwaitBounce) {
      ++returns;
    }
    last_stage = m.stage;
    last_striker = m.striker;
  }
  return chances > 0 ? static_cast<double>(returns) / chances : 0.0;
}

}  // namespace

TEST_CASE("magnet strength: only near the ball, only just before it arrives", "[magnet]") {
  const MagnetConfig cfg;
  // Close and imminent: full strength.
  CHECK(magnet_strength(cfg, 0.10f, 0.05f) == cfg.max_pull);
  // Too far away: nothing, however soon the ball arrives.
  CHECK(magnet_strength(cfg, 0.80f, 0.05f) == 0.0f);
  // The ball is still far off: nothing, however close the paddle is.
  CHECK(magnet_strength(cfg, 0.10f, 0.80f) == 0.0f);
  // In between, it grows as the ball approaches and as the paddle gets closer.
  CHECK(magnet_strength(cfg, 0.10f, 0.20f) > magnet_strength(cfg, 0.10f, 0.40f));
  CHECK(magnet_strength(cfg, 0.35f, 0.05f) > magnet_strength(cfg, 0.50f, 0.05f));

  MagnetConfig off;
  off.enabled = false;
  CHECK(magnet_strength(off, 0.10f, 0.05f) == 0.0f);
}

TEST_CASE("the magnet moves the paddle toward the ball but not its velocity", "[magnet]") {
  PlayerPaddle paddle;
  const Vec3 raw{0.20f, 1.00f, 1.50f}, meeting{0.0f, 0.90f, 1.50f};
  paddle.place(raw, {});
  paddle.set_meeting_point(meeting, 0.30f);
  for (int i = 0; i < 300; ++i) paddle.advance(kDt, kDt);  // the ball arrives

  CHECK(length(paddle.pos() - meeting) < 0.04f);  // was 22 cm away
  CHECK(paddle.raw_pos() == raw);                 // the input itself is untouched
  CHECK(length(paddle.vel()) < 0.01f);            // no invented swing speed

  // Once the ball has well and truly gone by, the paddle returns to the hand.
  for (int i = 0; i < 700; ++i) paddle.advance(kDt, kDt);
  CHECK(length(paddle.pos() - raw) < 0.005f);
}

TEST_CASE("a paddle in the wrong place is not rescued", "[magnet]") {
  PlayerPaddle paddle;
  const Vec3 raw{0.70f, 1.20f, 1.50f}, meeting{-0.10f, 0.90f, 1.50f};  // 85 cm away
  paddle.place(raw, {});
  paddle.set_meeting_point(meeting, 0.30f);
  for (int i = 0; i < 300; ++i) paddle.advance(kDt, kDt);
  CHECK(length(paddle.pos() - raw) < 0.01f);
}

TEST_CASE("how much the magnet helps a player who is only roughly on target", "[magnet][match]") {
  // Paddle lands within 30 cm of the right spot (the bat itself reaches 11).
  const double without = return_rate(false, 0.30f, 150);
  const double with = return_rate(true, 0.30f, 150);
  INFO("returned: " << without * 100 << "% without the magnet, " << with * 100 << "% with it");
  CHECK(without < 0.45);
  CHECK(with > 0.90);

  // Wildly off (up to 90 cm): the magnet must not turn that into a sure hit.
  const double wild = return_rate(true, 0.90f, 150);
  INFO("returned when up to 90 cm off, magnet on: " << wild * 100 << "%");
  CHECK(wild < 0.60);
}
