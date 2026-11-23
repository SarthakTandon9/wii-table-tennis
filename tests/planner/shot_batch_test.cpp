#include "planner/shot_batch.hpp"

#include "concurrency/thread_pool.hpp"
#include "game/shot.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

using namespace pp;
using Catch::Matchers::WithinAbs;

namespace {

// A spread of plausible strokes from the player's end: different directions,
// paces, lofts and spins. Some land, some go long, some hit the net.
std::vector<ShotCandidate> make_candidates(std::size_t n) {
  std::vector<ShotCandidate> out(n);
  std::uint32_t state = 0xFEEDu;
  auto unit = [&] {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 16777216.0f;
  };
  for (std::size_t i = 0; i < out.size(); ++i) {
    ShotCandidate& c = out[i];
    c.vel = {-2.0f + 4.0f * unit(), -1.0f + 4.5f * unit(), -(3.0f + 9.0f * unit())};
    c.spin = {-200.0f + 400.0f * unit(), -60.0f + 120.0f * unit(), -40.0f + 80.0f * unit()};
    if (i % 40 == 7) c.vel.y = 9.0f;  // a sky-high lob: still in the air when time runs out
  }
  return out;
}

ShotQuery make_query() {
  ShotQuery q;
  q.from = {0.2f, 0.95f, 1.45f};
  return q;
}

using Simulate = void (*)(const ShotQuery&, ShotBatch&, std::size_t, std::size_t);

void check_matches_reference(Simulate simulate, const char* name) {
  const ShotQuery q = make_query();
  const auto candidates = make_candidates(1003);  // not a multiple of 4, nor of the block size
  std::vector<ShotOutcome> reference(candidates.size());
  simulate_aos(q, candidates, reference);

  ShotBatch batch;
  batch.resize(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) batch.set(i, candidates[i]);
  simulate(q, batch, 0, batch.padded());

  int landed = 0, never = 0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    INFO(name << ", candidate " << i);
    const ShotOutcome got = batch.outcome(i), want = reference[i];
    // Vector code may round the last bit differently; results agree to well
    // under a millimetre.
    REQUIRE_THAT(got.time, WithinAbs(static_cast<double>(want.time), 1e-4));
    REQUIRE_THAT(got.land_x, WithinAbs(static_cast<double>(want.land_x), 2e-4));
    REQUIRE_THAT(got.land_z, WithinAbs(static_cast<double>(want.land_z), 2e-4));
    REQUIRE_THAT(got.clearance, WithinAbs(static_cast<double>(want.clearance), 2e-4));
    (want.time >= 0 ? landed : never) += 1;
  }
  CHECK(landed > 500);  // the test set really does exercise both outcomes
  CHECK(never > 5);
}

}  // namespace

TEST_CASE("the reference agrees with the game's own flight model", "[planner]") {
  // simulate_aos is a re-implementation for speed. It must give the same
  // answer as the code the game actually plays with.
  const World world;
  ShotQuery q = make_query();
  q.land_y = world.table.height + world.ball_params.radius;
  q.net_top = world.table.height + world.net.height;
  const auto candidates = make_candidates(200);
  std::vector<ShotOutcome> out(candidates.size());
  simulate_aos(q, candidates, out);

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const Flight f = fly(world, {.pos = q.from, .vel = candidates[i].vel, .spin = candidates[i].spin},
                         static_cast<real>(q.max_steps) * q.dt);
    INFO("candidate " << i);
    REQUIRE((out[i].time >= 0) == f.landed);
    if (!f.landed) continue;
    CHECK_THAT(out[i].land_x, WithinAbs(static_cast<double>(f.landing.x), 1e-3));
    CHECK_THAT(out[i].land_z, WithinAbs(static_cast<double>(f.landing.z), 1e-3));
    CHECK_THAT(out[i].time, WithinAbs(static_cast<double>(f.time), 1e-3));
    if (f.crossed_net) {
      CHECK_THAT(out[i].clearance, WithinAbs(static_cast<double>(f.net_clearance), 1e-3));
    }
  }
}

TEST_CASE("every fast version gives the reference answer", "[planner]") {
  check_matches_reference(&simulate_soa_wide, "soa wide");
  check_matches_reference(&simulate_soa_blocked, "soa blocked");
  check_matches_reference(&simulate_neon, "neon");
}

TEST_CASE("thread pool: every index is visited exactly once", "[pool][threads]") {
  ThreadPool pool{3};
  CHECK(pool.threads() == 4);
  for (const std::size_t chunk : {std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{5000}}) {
    std::vector<std::atomic<int>> visits(1000);
    pool.parallel_for(visits.size(), chunk, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) visits[i].fetch_add(1);
    });
    int wrong = 0;
    for (const auto& v : visits) wrong += v.load() != 1;
    CHECK(wrong == 0);
  }
  pool.parallel_for(0, 4, [](std::size_t, std::size_t) { FAIL("called for an empty range"); });
}

TEST_CASE("thread pool: a batch split across threads matches one thread", "[pool][planner][threads]") {
  const ShotQuery q = make_query();
  const auto candidates = make_candidates(4096);
  ShotBatch serial, parallel;
  serial.resize(candidates.size());
  parallel.resize(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    serial.set(i, candidates[i]);
    parallel.set(i, candidates[i]);
  }
  simulate_neon(q, serial, 0, serial.padded());

  ThreadPool pool{3};
  pool.parallel_for(parallel.padded(), 256, [&](std::size_t begin, std::size_t end) {
    simulate_neon(q, parallel, begin, end);
  });

  // Threads write to disjoint ranges of the same arrays, and each candidate
  // is computed independently, so the result is bit-for-bit the same.
  CHECK(serial.time == parallel.time);
  CHECK(serial.land_x == parallel.land_x);
  CHECK(serial.land_z == parallel.land_z);
  CHECK(serial.clearance == parallel.clearance);
}
