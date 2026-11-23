#include "concurrency/spsc_ring.hpp"
#include "concurrency/triple_buffer.hpp"
#include "game/sim_thread.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

using namespace pp;
using namespace std::chrono_literals;

// The two-thread tests are only half the story on their own: a data race can
// pass a million times and fail on the next. Their real job is to give
// ThreadSanitizer something to watch. Run them with:
//   cmake --build --preset tsan && ctest --preset tsan

// ------------------------------------------------------------------ SpscRing

TEST_CASE("ring: first in, first out", "[spsc]") {
  SpscRing<int, 8> ring;
  int out = -1;
  CHECK_FALSE(ring.try_pop(out));  // starts empty

  for (int i = 0; i < 5; ++i) REQUIRE(ring.try_push(i));
  for (int i = 0; i < 5; ++i) {
    REQUIRE(ring.try_pop(out));
    CHECK(out == i);
  }
  CHECK_FALSE(ring.try_pop(out));
}

TEST_CASE("ring: holds exactly its capacity, then refuses", "[spsc]") {
  SpscRing<int, 4> ring;
  for (int i = 0; i < 4; ++i) REQUIRE(ring.try_push(i));
  CHECK_FALSE(ring.try_push(99));

  int out = -1;
  REQUIRE(ring.try_pop(out));
  CHECK(out == 0);
  CHECK(ring.try_push(4));  // one slot freed, one accepted
  CHECK_FALSE(ring.try_push(5));
}

TEST_CASE("ring: stays correct as the indices wrap around the buffer", "[spsc]") {
  SpscRing<int, 4> ring;
  int out = -1, next_in = 0, next_out = 0;
  for (int round = 0; round < 1000; ++round) {
    for (int k = 0; k < 3; ++k) REQUIRE(ring.try_push(next_in++));
    for (int k = 0; k < 3; ++k) {
      REQUIRE(ring.try_pop(out));
      REQUIRE(out == next_out++);
    }
  }
}

TEST_CASE("ring: two threads, nothing lost, nothing reordered", "[spsc][threads]") {
  constexpr std::uint64_t kCount = 2'000'000;
  SpscRing<std::uint64_t, 1024> ring;

  std::thread producer([&] {
    for (std::uint64_t i = 1; i <= kCount; ++i) {
      while (!ring.try_push(i)) std::this_thread::yield();
    }
  });

  std::uint64_t expected = 1, sum = 0, value = 0;
  while (expected <= kCount) {
    if (!ring.try_pop(value)) {
      std::this_thread::yield();
      continue;
    }
    if (value != expected) break;  // fail outside the hot loop
    sum += value;
    ++expected;
  }
  producer.join();

  CHECK(expected == kCount + 1);
  CHECK(sum == kCount * (kCount + 1) / 2);
}

// -------------------------------------------------------------- TripleBuffer

TEST_CASE("triple buffer: reader sees the newest value, repeatedly", "[triple]") {
  TripleBuffer<int> buffer;
  CHECK(buffer.latest() == 0);  // default value before anything is published

  buffer.publish(1);
  buffer.publish(2);
  buffer.publish(3);  // 1 and 2 are overwritten unseen: latest wins
  CHECK(buffer.latest() == 3);
  CHECK(buffer.latest() == 3);  // reading again does not go back to a stale slot

  buffer.publish(4);
  CHECK(buffer.latest() == 4);
}

namespace {

// A value that is only valid if all of it was written together.
struct Record {
  std::uint64_t id = 0;
  std::array<std::uint64_t, 31> copies{};

  static Record make(std::uint64_t id) {
    Record r;
    r.id = id;
    r.copies.fill(id * 0x9E3779B97F4A7C15ull);
    return r;
  }
  bool intact() const {
    for (const std::uint64_t c : copies) {
      if (c != id * 0x9E3779B97F4A7C15ull) return false;
    }
    return true;
  }
};

}  // namespace

TEST_CASE("triple buffer: reader never sees a torn or older value", "[triple][threads]") {
  TripleBuffer<Record> buffer;
  std::atomic<bool> done{false};
  constexpr std::uint64_t kWrites = 1'000'000;

  std::thread writer([&] {
    for (std::uint64_t i = 1; i <= kWrites; ++i) buffer.publish(Record::make(i));
    done = true;
  });

  std::uint64_t last = 0, reads = 0;
  bool torn = false, went_back = false;
  for (;;) {
    const bool writer_finished = done.load();
    const Record& r = buffer.latest();
    torn |= !r.intact();
    went_back |= r.id < last;
    last = r.id;
    ++reads;
    if (writer_finished) break;  // read once more after the last publish
  }
  writer.join();

  CHECK_FALSE(torn);
  CHECK_FALSE(went_back);
  CHECK(buffer.latest().id == kWrites);  // the final value always arrives
  CHECK(reads > 0);
}

// ----------------------------------------------------------------- SimThread

namespace {

double test_clock() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

TEST_CASE("sim thread: runs at the requested speed and stops cleanly", "[sim][threads]") {
  SimThread sim{&test_clock};
  REQUIRE(sim.send({.kind = SimCommand::Kind::SetSpeed, .value = 1.0f}));
  std::this_thread::sleep_for(50ms);  // let the speed change take effect

  const std::uint64_t start_tick = sim.latest().tick;
  const auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(400ms);
  const std::uint64_t ticks = sim.latest().tick - start_tick;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  // 1000 ticks per second of wall time at speed 1. Generous bounds: this must
  // also pass under sanitizers and on a loaded machine.
  CHECK(static_cast<double>(ticks) > 1000.0 * elapsed * 0.80);
  CHECK(static_cast<double>(ticks) < 1000.0 * elapsed * 1.10);
}  // destructor joins the thread; a hang here is a failure

TEST_CASE("sim thread: commands take effect and the paddle follows the pointer", "[sim][threads]") {
  SimThread sim{&test_clock};
  sim.send({.kind = SimCommand::Kind::SetAssist, .flag = false});
  sim.send({.kind = SimCommand::Kind::SetSpeed, .value = 0.5f});
  const Vec3 target{0.4f, 1.2f, 1.38f};
  for (int i = 0; i < 20; ++i) {
    sim.send({.kind = SimCommand::Kind::PointerTarget, .target = target});
    std::this_thread::sleep_for(10ms);
  }

  const Snapshot& snap = sim.latest();
  CHECK_FALSE(snap.assist);
  CHECK(snap.speed == 0.5f);
  const Vec3 paddle = snap.paddles[index(Side::Player)].pos;
  CHECK(std::abs(paddle.x - target.x) < 0.01f);
  CHECK(std::abs(paddle.y - target.y) < 0.01f);
}

TEST_CASE("sim thread: hand observations drive the paddle when enabled", "[sim][threads]") {
  SimThread sim{&test_clock};
  sim.send({.kind = SimCommand::Kind::UseHand, .flag = true});

  // An open hand, re-observed at 100 Hz. It first rests in one place (which
  // becomes the paddle's rest position), then moves to the right.
  const auto hand_at = [](float x) {
    HandObservation obs;
    obs.found = true;
    obs.aspect = 4.0f / 3.0f;
    for (JointPoint& j : obs.joints) j = {.x = x, .y = 0.50f, .confidence = 0.9f};
    obs.joints[static_cast<std::size_t>(HandJoint::Wrist)].y = 0.40f;
    obs.joints[static_cast<std::size_t>(HandJoint::IndexMCP)].x = x - 0.03f;  // a knuckle row with width
    obs.joints[static_cast<std::size_t>(HandJoint::LittleMCP)].x = x + 0.03f;
    obs.capture_time = test_clock();
    obs.detect_time = obs.capture_time;
    return obs;
  };

  std::thread camera([&] {  // push_hand must come from one thread: this one
    for (int i = 0; i < 60; ++i) {
      // Unhurried: 12% of the image over 0.3 s. (A sudden jump would be read
      // as a swing, and a swing deliberately does not move the bat.)
      sim.push_hand(hand_at(0.50f + 0.12f * static_cast<float>(std::clamp(i - 25, 0, 30)) / 30.0f));
      std::this_thread::sleep_for(10ms);
    }
  });
  camera.join();

  const Snapshot& snap = sim.latest();
  CHECK(snap.hands_received == 60);
  CHECK(snap.hands_dropped == 0);
  CHECK(snap.hand_in_control);
  CHECK(snap.paddles[index(Side::Player)].pos.x > 0.3f);  // hand moved right, paddle went right
}
