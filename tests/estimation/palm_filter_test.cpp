#include "estimation/palm_filter.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace pp;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kFrame = 1.0 / 30.0;  // camera interval
constexpr double kDelay = 0.070;       // exposure -> usable, as measured on the real pipeline

// An upright open hand whose palm centre is exactly (cx, cy) and whose
// apparent size scales with `length`.
HandObservation hand_at(double t, float cx, float cy, float length = 0.10f) {
  HandObservation obs;
  obs.found = true;
  obs.aspect = 4.0f / 3.0f;
  obs.capture_time = t;
  obs.detect_time = t + kDelay;
  for (JointPoint& j : obs.joints) j = {.x = cx, .y = cy, .confidence = 0.9f};
  auto set = [&](HandJoint j, float dx, float dy) {
    obs.joints[static_cast<std::size_t>(j)] = {cx + dx * length, cy + dy * length, 0.9f};
  };
  set(HandJoint::Wrist, 0.0f, -0.8f);
  set(HandJoint::IndexMCP, -0.3f, 0.2f);
  set(HandJoint::MiddleMCP, -0.1f, 0.2f);
  set(HandJoint::RingMCP, 0.1f, 0.2f);
  set(HandJoint::LittleMCP, 0.3f, 0.2f);
  return obs;
}

// Repeatable noise, roughly bell-shaped, standard deviation `sigma`.
struct Noise {
  std::uint32_t state = 0x1234567u;
  float next(float sigma) {
    float sum = 0;
    for (int i = 0; i < 4; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      sum += static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
    }
    return sum * sigma * 1.7320508f;  // 4 uniforms of variance 1/12 each
  }
};

// A fixed mapping for these tests: the middle of the image is the rest
// position and 0.6 of the image spans the paddle range. (In the game the
// centre is learned from where the hand first appears; that has its own test.)
PalmFilterConfig fixed_mapping() {
  PalmFilterConfig c;
  c.learn_centre = false;
  c.span_x = 0.6f;
  c.span_y = 0.6f;
  c.rest_v = 0.5f;
  c.pos_noise = 0.0045f;  // these tests feed noise of about this size
  c.max_lookahead = 0.11;  // the whole camera delay: these tests are about the prediction itself
  return c;
}

// Image x -> world x, the same mapping the filter applies.
float world_x(const PalmFilterConfig& c, float image_x) {
  return (2.0f * (0.5f + (image_x - c.centre_x) / c.span_x) - 1.0f) * c.world_half_width;
}

double stddev(const std::vector<double>& v) {
  double mean = 0;
  for (const double x : v) mean += x;
  mean /= static_cast<double>(v.size());
  double var = 0;
  for (const double x : v) var += (x - mean) * (x - mean);
  return std::sqrt(var / static_cast<double>(v.size()));
}

}  // namespace

TEST_CASE("a still hand: the paddle shakes far less than the raw measurement", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  Noise noise;
  const float sigma = 0.004f;  // about 2.5 pixels of a 640-wide image
  std::vector<double> raw, filtered;

  for (int frame = 0; frame < 150; ++frame) {
    const double t = frame * kFrame;
    const float x = 0.5f + noise.next(sigma);
    filter.observe(hand_at(t, x, 0.5f + noise.next(sigma)));
    if (frame < 30) continue;  // let it settle
    raw.push_back(static_cast<double>(world_x(filter.config(), x)));
    // What the game uses: the pose predicted to "now", delay included.
    filtered.push_back(static_cast<double>(filter.pose_at(t + kDelay).pos.x));
  }

  const double raw_mm = 1000 * stddev(raw), filtered_mm = 1000 * stddev(filtered);
  INFO("jitter: raw " << raw_mm << " mm, filtered " << filtered_mm << " mm");
  // Roughly halved. It cannot go much lower without making the paddle slow
  // to follow a real movement; the sweep in palm_filter.hpp's comment chose
  // this point.
  CHECK(filtered_mm < 0.60 * raw_mm);
}

TEST_CASE("a moving hand: prediction cancels the camera delay", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  Noise noise;
  const float speed = 0.5f;  // image widths per second: a brisk sweep, 1.25 m/s in the world
  double raw_error = 0, filtered_error = 0;
  int samples = 0;

  for (int frame = 0; frame < 30; ++frame) {
    const double t = frame * kFrame;
    const float x = 0.25f + speed * static_cast<float>(t);
    filter.observe(hand_at(t, x + noise.next(0.003f), 0.5f));
    if (frame < 8) continue;

    // The game sees this frame kDelay later. Where is the hand really, then?
    const double now = t + kDelay;
    const float truth = world_x(filter.config(), 0.25f + speed * static_cast<float>(now));
    raw_error += std::abs(static_cast<double>(world_x(filter.config(), x) - truth));
    filtered_error += std::abs(static_cast<double>(filter.pose_at(now).pos.x - truth));
    ++samples;
  }
  raw_error /= samples;
  filtered_error /= samples;

  INFO("lag at 1.25 m/s: raw " << raw_error * 100 << " cm, filtered " << filtered_error * 100
                               << " cm");
  CHECK(raw_error > 0.08);              // unfiltered: 70 ms behind is almost 9 cm
  CHECK(filtered_error < 0.010);        // predicted: within 1 cm
  CHECK_THAT(filter.pose_at(29 * kFrame + kDelay).vel.x, WithinAbs(1.25, 0.20));
}

TEST_CASE("a sudden swing is followed within a few frames", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  for (int frame = 0; frame < 30; ++frame) filter.observe(hand_at(frame * kFrame, 0.3f, 0.5f));

  // From rest, the hand takes off at 1 image width per second (2.5 m/s).
  int frames_to_catch_up = -1;
  for (int frame = 1; frame <= 12; ++frame) {
    const double t = (29 + frame) * kFrame;
    const float x = 0.3f + 1.0f * static_cast<float>(frame * kFrame);
    filter.observe(hand_at(t, x, 0.5f));
    const double now = t + kDelay;
    const float truth =
        world_x(filter.config(), 0.3f + 1.0f * static_cast<float>(frame * kFrame + kDelay));
    if (std::abs(filter.pose_at(now).pos.x - truth) < 0.04f) {
      frames_to_catch_up = frame;
      break;
    }
  }
  INFO("frames to get within 4 cm of a hand that suddenly moves at 2.5 m/s: "
       << frames_to_catch_up);
  REQUIRE(frames_to_catch_up > 0);
  CHECK(frames_to_catch_up <= 5);  // about 170 ms
}

TEST_CASE("between camera frames the paddle moves smoothly, not in steps", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  Noise noise;
  const float speed = 0.5f;
  float previous = 0, biggest_jump = 0;
  bool have_previous = false;

  for (int frame = 0; frame < 40; ++frame) {
    const double t = frame * kFrame;
    filter.observe(hand_at(t, 0.25f + speed * static_cast<float>(t) + noise.next(0.003f), 0.5f));
    if (frame < 10) continue;
    // The simulation asks a thousand times a second.
    for (int ms = 0; ms < 33; ++ms) {
      const float x = filter.pose_at(t + kDelay + ms * 0.001).pos.x;
      if (have_previous) biggest_jump = std::max(biggest_jump, std::abs(x - previous));
      previous = x;
      have_previous = true;
    }
  }
  // Raw samples would step 4 cm, 30 times a second. 1.25 m/s is 1.25 mm per
  // millisecond; allow the small correction when a new frame arrives.
  INFO("largest 1 ms jump: " << biggest_jump * 1000 << " mm");
  CHECK(biggest_jump < 0.012f);
}

TEST_CASE("depth: a hand that looks bigger is reaching forward", "[filter][depth]") {
  // The game keeps depth small (see PalmFilterConfig); a wide range is set
  // here so the mechanism itself can be checked.
  PalmFilterConfig wide = fixed_mapping();
  wide.depth_gain = 0.9f;
  wide.max_forward = 0.40f;
  wide.max_back = 0.15f;
  PalmFilter filter{wide};
  const PalmFilterConfig cfg = filter.config();
  double t = 0;
  for (int frame = 0; frame < 30; ++frame, t += kFrame) filter.observe(hand_at(t, 0.5f, 0.5f));
  CHECK_THAT(filter.pose_at(t).pos.z, WithinAbs(static_cast<double>(cfg.base_z), 0.01));

  // 20% bigger in the image: the hand moved from, say, 60 cm to 50 cm away.
  for (int frame = 0; frame < 45; ++frame, t += kFrame) {
    filter.observe(hand_at(t, 0.5f, 0.5f, 0.12f));
  }
  const double expected = static_cast<double>(cfg.base_z - cfg.depth_gain * std::log(1.2f));
  CHECK_THAT(filter.pose_at(t).pos.z, WithinAbs(expected, 0.02));
  CHECK(filter.pose_at(t).pos.z < cfg.base_z - 0.12f);  // toward the table

  // Pulled far back: limited, so the paddle cannot leave through the wall.
  for (int frame = 0; frame < 60; ++frame, t += kFrame) {
    filter.observe(hand_at(t, 0.5f, 0.5f, 0.05f));
  }
  CHECK_THAT(filter.pose_at(t).pos.z, WithinAbs(static_cast<double>(cfg.base_z + cfg.max_back), 1e-4));
}

TEST_CASE("depth is not fooled by the hand tilting", "[filter][depth]") {
  // Tilting the hand forward shortens the wrist-to-knuckle length in the
  // image but leaves the knuckle row as wide as before.
  HandObservation flat = hand_at(0, 0.5f, 0.5f);
  HandObservation tilted = flat;
  tilted.joints[static_cast<std::size_t>(HandJoint::Wrist)].y += 0.04f;  // length nearly halved
  CHECK_THAT(apparent_hand_size(tilted), WithinAbs(static_cast<double>(apparent_hand_size(flat)), 1e-5));
}

TEST_CASE("the rest position is learned from where the hand first appears", "[filter]") {
  // The real case from recorded play: the hand low in the frame and off to
  // the right. Wherever it starts must become the paddle's rest position.
  PalmFilterConfig no_drift;
  no_drift.centre_follow_seconds = 0;  // the slow drift has its own test below
  PalmFilter filter{no_drift};
  const PalmFilterConfig cfg = filter.config();
  double t = 0;
  for (int frame = 0; frame < 30; ++frame, t += kFrame) filter.observe(hand_at(t, 0.58f, 0.27f));

  const HandPose rest = filter.pose_at(t);
  REQUIRE(rest.valid);
  CHECK_THAT(rest.pos.x, WithinAbs(0.0, 0.01));  // centred, not 14 cm to the right
  const double rest_height = static_cast<double>(
      cfg.world_min_y + cfg.rest_v * (cfg.world_max_y - cfg.world_min_y));
  CHECK_THAT(rest.pos.y, WithinAbs(rest_height, 0.01));

  // From there, a modest movement of the hand reaches the edge of the range.
  for (int frame = 0; frame < 30; ++frame, t += kFrame) {
    filter.observe(hand_at(t, 0.58f + cfg.span_x / 2, 0.27f));
  }
  CHECK_THAT(filter.pose_at(t).pos.x, WithinAbs(static_cast<double>(cfg.world_half_width), 0.02));

  // The hand's own speed is reported in image units, whatever the mapping.
  for (int frame = 0; frame < 20; ++frame, t += kFrame) {
    filter.observe(hand_at(t, 0.5f + 0.9f * static_cast<float>(frame * kFrame), 0.27f));
  }
  CHECK_THAT(filter.pose_at(t).image_vel.x, WithinAbs(0.9, 0.15));
}

TEST_CASE("one wild frame is ignored; a hand that really moved is believed", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  double t = 0;
  for (int frame = 0; frame < 20; ++frame, t += kFrame) filter.observe(hand_at(t, 0.40f, 0.5f));
  const float before = filter.pose_at(t).pos.x;

  // The detector briefly latches onto something else, half the image away.
  filter.observe(hand_at(t, 0.85f, 0.5f));
  t += kFrame;
  const HandPose after_glitch = filter.pose_at(t);
  CHECK_THAT(after_glitch.pos.x, WithinAbs(static_cast<double>(before), 0.01));
  CHECK(length(after_glitch.image_vel) < 0.2f);  // and it is not mistaken for a swing

  // Back to normal: tracking carries on as if nothing happened.
  for (int frame = 0; frame < 5; ++frame, t += kFrame) filter.observe(hand_at(t, 0.40f, 0.5f));
  CHECK_THAT(filter.pose_at(t).pos.x, WithinAbs(static_cast<double>(before), 0.01));

  // If the hand genuinely is somewhere new (it left the frame and came back
  // elsewhere), a few consistent frames are enough to follow it.
  for (int frame = 0; frame < 6; ++frame, t += kFrame) filter.observe(hand_at(t, 0.70f, 0.5f));
  CHECK(filter.pose_at(t).pos.x > before + 0.5f);
}

TEST_CASE("a fast swing is tracked, not rejected as a glitch", "[filter]") {
  // From recorded play: sweeps of 3.4 image widths per second were being
  // thrown away as camera errors. That is 0.11 of the image per frame.
  PalmFilter filter{fixed_mapping()};
  double t = 0;
  for (int frame = 0; frame < 20; ++frame, t += kFrame) filter.observe(hand_at(t, 0.30f, 0.5f));
  float peak = 0;
  for (int frame = 1; frame <= 5; ++frame, t += kFrame) {
    filter.observe(hand_at(t, 0.30f + 0.11f * static_cast<float>(frame), 0.5f));
    peak = std::max(peak, filter.pose_at(t).image_vel.x);
  }
  CHECK(peak > 3.0f);
  CHECK(filter.pose_at(t).pos.x > 0.5f);  // and the bat went with it
}

TEST_CASE("a push toward the camera reads as speed; a turning hand does not", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  double t = 0;
  for (int frame = 0; frame < 20; ++frame, t += kFrame) filter.observe(hand_at(t, 0.5f, 0.5f));

  // The whole hand grows 8% a frame: it is coming closer.
  float push = 0, size = 0.10f;
  for (int frame = 0; frame < 5; ++frame, t += kFrame) {
    size *= 1.08f;
    filter.observe(hand_at(t, 0.5f, 0.5f, size));
    push = std::max(push, filter.pose_at(t).image_vel.z);
  }
  CHECK(push > 1.8f);  // ln(1.08) * 30 frames/s = 2.3 per second

  // Tilting forward: the palm looks shorter then longer, but no wider.
  PalmFilter turning{fixed_mapping()};
  t = 0;
  float turn = 0;
  for (int frame = 0; frame < 30; ++frame, t += kFrame) {
    HandObservation obs = hand_at(t, 0.5f, 0.5f);
    const float lengthen = frame < 20 ? 0.0f : 0.012f * static_cast<float>(frame - 20);
    obs.joints[static_cast<std::size_t>(HandJoint::Wrist)].y -= lengthen;
    turning.observe(obs);
    if (frame >= 20) turn = std::max(turn, turning.pose_at(t).image_vel.z);
  }
  CHECK(turn < 0.3f);
}

TEST_CASE("a palm with a hidden landmark is still tracked", "[filter]") {
  // The wrist drops out of the bottom of the picture; the knuckles remain.
  PalmFilter filter{fixed_mapping()};
  double t = 0;
  for (int frame = 0; frame < 20; ++frame, t += kFrame) filter.observe(hand_at(t, 0.40f, 0.5f));
  for (int frame = 1; frame <= 15; ++frame, t += kFrame) {
    HandObservation obs = hand_at(t, 0.40f + 0.01f * static_cast<float>(frame), 0.5f);
    obs.joints[static_cast<std::size_t>(HandJoint::Wrist)].confidence = 0.05f;
    filter.observe(obs);
  }
  const HandPose pose = filter.pose_at(t - kFrame + 0.03);
  REQUIRE(pose.valid);
  CHECK_THAT(pose.pos.x, WithinAbs(static_cast<double>(world_x(filter.config(), 0.55f)), 0.05));
}

TEST_CASE("the rest position slowly follows where the hand settles", "[filter]") {
  PalmFilter filter;
  double t = 0;
  for (int frame = 0; frame < 30; ++frame, t += kFrame) filter.observe(hand_at(t, 0.70f, 0.3f));
  // The hand moves over and stays there.
  for (int frame = 0; frame < 15; ++frame, t += kFrame) filter.observe(hand_at(t, 0.60f, 0.3f));
  const float just_moved = filter.pose_at(t).pos.x;
  CHECK(just_moved < -0.3f);  // at first that is a real move to the left

  // It does not snap back: half a minute on, the bat is still left of centre.
  for (int frame = 0; frame < 30 * 25; ++frame, t += kFrame) filter.observe(hand_at(t, 0.60f, 0.3f));
  CHECK(filter.pose_at(t).pos.x < -0.15f);
  // But a hand that stays put for good becomes the new centre.
  for (int frame = 0; frame < 30 * 90; ++frame, t += kFrame) filter.observe(hand_at(t, 0.60f, 0.3f));
  CHECK(std::abs(filter.pose_at(t).pos.x) < 0.08f);
}

TEST_CASE("the sideways range fits itself to how far the hand actually moves", "[filter]") {
  // A player who moves the hand only 10% of the picture, side to side.
  const auto sway = [](PalmFilter& filter, float half_range, double seconds) {
    float leftmost = 0, rightmost = 0;
    double t = 0;
    for (; t < seconds; t += kFrame) {
      const float x = 0.60f + half_range * std::sin(static_cast<float>(t) * 1.2f);
      filter.observe(hand_at(t, x, 0.3f));
      if (t > seconds - 6.0) {  // the last full sway
        leftmost = std::min(leftmost, filter.pose_at(t).pos.x);
        rightmost = std::max(rightmost, filter.pose_at(t).pos.x);
      }
    }
    return std::pair{leftmost, rightmost};
  };
  PalmFilter small_mover, big_mover;
  const auto [small_left, small_right] = sway(small_mover, 0.05f, 90.0);
  const auto [big_left, big_right] = sway(big_mover, 0.20f, 90.0);
  // Both reach well out to both sides of the table (half-width 0.75 m)...
  CHECK(small_left < -0.35f);
  CHECK(small_right > 0.35f);
  CHECK(big_left < -0.60f);
  CHECK(big_right > 0.60f);
  // ...although one moved the hand four times as far as the other.
}

TEST_CASE("the filter coasts through a short dropout and gives up after a long one", "[filter]") {
  PalmFilter filter{fixed_mapping()};
  double t = 0;
  for (int frame = 0; frame < 20; ++frame, t += kFrame) {
    filter.observe(hand_at(t, 0.3f + 0.5f * static_cast<float>(t), 0.5f));
  }
  const double last = t - kFrame;

  // Frames with no usable hand do no harm.
  HandObservation nothing;
  nothing.capture_time = t;
  filter.observe(nothing);

  CHECK(filter.pose_at(last + 0.15).valid);        // a few missed frames: keep going
  CHECK_FALSE(filter.pose_at(last + 0.40).valid);  // gone: hand control ends

  // It does not extrapolate without limit while coasting.
  const float at_cap = filter.pose_at(last + 0.11).pos.x;
  CHECK(filter.pose_at(last + 0.25).pos.x == at_cap);
}
