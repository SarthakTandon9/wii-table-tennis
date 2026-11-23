#include "game/telemetry.hpp"

#include "game/paddle_surface.hpp"
#include "game/session_log.hpp"
#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <vector>

using namespace pp;

namespace {

constexpr real kDt = 1e-3f;

// Plays points with a paddle that goes to the meeting point plus a fixed
// offset, and collects what the tracker records.
struct Recorded {
  std::vector<ChanceRecord> chances;
  std::vector<PointRecord> points;
};

Recorded play(Vec3 aim_offset, int max_chances) {
  Match m{Side::Opponent, 0x5EEDu};
  PlayerPaddle paddle;
  paddle.magnet.enabled = false;  // measure the raw aim, unassisted
  TelemetryTracker tracker;
  Recorded out;
  double now = 100.0;

  for (int i = 0; i < 2'000'000 && static_cast<int>(out.chances.size()) < max_chances; ++i) {
    if (m.phase == Phase::GameOver) m = Match{Side::Opponent, 0x5EEDu + static_cast<unsigned>(i)};

    std::optional<Vec3> meeting;
    if (i % 4 == 0 && m.phase == Phase::Rally && m.striker == Side::Opponent) {
      const auto c = predict_crossing(m.world, {0, 0, 1.50f}, {0, 0, -1},
                                      m.stage == Stage::AwaitHit);
      if (c && c->bounced) {
        meeting = c->point;
        paddle.place(c->point + aim_offset, {});
      }
    }
    const Vec3 at = paddle.pos();
    tick(m, {.pos = at, .vel = paddle.vel(), .normal = synthetic_face_normal(at)}, kDt);
    paddle.advance(kDt, kDt);
    now += static_cast<double>(kDt);

    if (const auto c = tracker.update(m, paddle, {.now = now, .sim_speed = 1, .hand = false,
                                                  .meeting = meeting, .last_swing_start = 0})) {
      out.chances.push_back(*c);
    }
    if (const auto p = tracker.point(m, now)) out.points.push_back(*p);
  }
  return out;
}

}  // namespace

TEST_CASE("a player on target: chances are recorded as hits with a small aim error", "[telemetry]") {
  const Recorded r = play({}, 40);
  REQUIRE(r.chances.size() == 40);
  int hits = 0, good = 0;
  for (const ChanceRecord& c : r.chances) {
    hits += c.hit;
    good += c.result == ShotResult::Good;
    CHECK(c.miss_raw < 0.02f);
    CHECK(c.notice > 0.2f);        // the ball takes a while to arrive
    CHECK(c.ball_speed > 1.0f);
    CHECK_FALSE(c.hand);
  }
  CHECK(hits == 40);
  CHECK(good >= 36);  // the assisted return almost always lands
}

TEST_CASE("a player who is 40 cm off: chances are recorded as misses, with the offset", "[telemetry]") {
  const Recorded r = play({0.40f, 0, 0}, 20);
  REQUIRE(r.chances.size() == 20);
  for (const ChanceRecord& c : r.chances) {
    CHECK_FALSE(c.hit);
    CHECK(c.result == ShotResult::None);
    // The record is frozen as the ball passes: the offset is what it was at
    // arrival, to the centimetre.
    CHECK(std::abs(c.miss_raw - 0.40f) < 0.01f);
    CHECK(std::abs((c.raw.x - c.meeting.x) - 0.40f) < 0.01f);
    CHECK(c.settled_for < 0);  // never within 15 cm
  }
  // Every one of those balls cost a point. (The player still wins the odd
  // point when the opponent hits out; those are not chances.)
  int lost = 0;
  for (const PointRecord& p : r.points) lost += p.winner == Side::Opponent;
  CHECK(lost >= 19);
}

TEST_CASE("a session written to disk reads back exactly", "[telemetry][log]") {
  const auto root = std::filesystem::temp_directory_path() / "palmrally_test_sessions";
  std::filesystem::remove_all(root);

  HandObservation hand;
  hand.found = true;
  hand.capture_time = 12.5;
  hand.joints[3] = {.x = 0.25f, .y = 0.75f, .confidence = 0.5f};
  ChanceRecord chance;
  chance.hit = true;
  chance.miss_raw = 0.123f;
  chance.result = ShotResult::Net;
  PointRecord point;
  point.winner = Side::Opponent;
  point.reason = PointReason::DoubleBounce;
  point.rally_hits = 7;

  std::filesystem::path dir;
  {
    SessionLog log;
    REQUIRE(log.open(root));
    dir = log.directory();
    log.write({.kind = TelemetryEvent::Kind::Hand, .hand = hand, .chance = {}, .point = {}, .swing = {}});
    log.write({.kind = TelemetryEvent::Kind::Chance, .hand = {}, .chance = chance, .point = {}, .swing = {}});
    log.write({.kind = TelemetryEvent::Kind::Point, .hand = {}, .chance = {}, .point = point, .swing = {}});
  }  // closed here

  CHECK(latest_session(root) == dir);
  const auto data = load_session(dir);
  REQUIRE(data.has_value());
  REQUIRE(data->hands.size() == 1);
  REQUIRE(data->chances.size() == 1);
  REQUIRE(data->points.size() == 1);
  CHECK(data->hands[0].capture_time == 12.5);
  CHECK(data->hands[0].joints[3].y == 0.75f);
  CHECK(data->chances[0].miss_raw == 0.123f);
  CHECK(data->chances[0].result == ShotResult::Net);
  CHECK(data->points[0].reason == PointReason::DoubleBounce);
  CHECK(data->points[0].rally_hits == 7);

  CHECK_FALSE(load_session(root / "does-not-exist").has_value());
  std::filesystem::remove_all(root);
}
