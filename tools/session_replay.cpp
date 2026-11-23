// Replays a recorded session's camera frames through the game's own hand
// control (game/hand_control.hpp) and measures how the bat would behave.
//
//   session_replay [session directory]      (default: the newest session)
//
// The point: questions like "how far behind the hand is the bat?" or "how
// much does the bat shake?" can be answered for a change to the code from
// play that has already been recorded, before anyone has to play again.
// Frames are fed at the moment the game really received them, and the control
// runs a thousand times a second, as it does in the game.
//
// What is and is not replayed: the hand is real; the ball is not re-simulated.
// Each recorded ball contributes where and when it arrived, and from that the
// body and the magnet are driven as the game would drive them.
#include "estimation/palm.hpp"
#include "game/hand_control.hpp"
#include "game/match.hpp"
#include "game/player_paddle.hpp"
#include "game/session_log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace pp;

namespace {

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1))];
}

struct Sample {
  double time;
  double value;
};

// Linear interpolation in a time series; false outside it or across a gap.
bool at(const std::vector<Sample>& s, double t, double& out) {
  const auto hi = std::lower_bound(s.begin(), s.end(), t,
                                   [](const Sample& a, double time) { return a.time < time; });
  if (hi == s.begin() || hi == s.end()) return false;
  const auto lo = hi - 1;
  if (hi->time - lo->time > 0.07) return false;  // a dropped frame: no reference here
  const double u = (t - lo->time) / (hi->time - lo->time);
  out = lo->value + (hi->value - lo->value) * u;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path dir;
  if (argc > 1) {
    dir = argv[1];
  } else if (const auto latest = latest_session(SessionLog::default_root())) {
    dir = *latest;
  }
  const auto data = load_session(dir);
  if (!data || data->hands.empty()) {
    std::printf("cannot read a session from %s\n", dir.c_str());
    return 1;
  }
  std::printf("replaying %s: %zu camera frames, %zu balls\n\n", dir.c_str(), data->hands.size(),
              data->chances.size());

  // Optional overrides, for trying settings: lookahead=0.07 cutoff=5 beta=3
  PalmFilterConfig filter_config = arm_filter_config();
  BatSmootherConfig smoothing = kArmSmoothing;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    const auto value = [&](std::string_view key, auto& into) {
      if (arg.starts_with(key)) into = static_cast<std::remove_reference_t<decltype(into)>>(std::atof(arg.substr(key.size()).data()));
    };
    value("lookahead=", filter_config.max_lookahead);
    value("cutoff=", smoothing.min_cutoff);
    value("beta=", smoothing.beta);
  }
  HandControl control{{}, filter_config, smoothing};
  PlayerPaddle paddle;
  paddle.magnet = kHandMagnet;
  const real speed = data->chances.empty() ? real{0.8f} : data->chances.front().sim_speed;

  std::vector<const ChanceRecord*> balls;
  for (const ChanceRecord& c : data->chances) {
    if (c.hand) balls.push_back(&c);
  }

  // Series for the lag measurement: where the hand truly was (measured, at
  // the time the camera saw it) against where the bat was drawn.
  std::vector<Sample> truth, drawn;
  std::vector<double> shake, moved_by_game, body_gap, contact_speed, swing_travel;
  int returned = 0, counted = 0, anticipated = 0;
  int missed_no_swing = 0, missed_sideways = 0, missed_height = 0;
  bool swung_near = false;
  real nearest_side = 1e9f, nearest_height = 1e9f;

  std::size_t frame = 0, ball = 0;
  bool have_prev = false, ball_hit = false;
  Vec3 prev_arm{}, prev_step{};
  Vec3 swing_from{};
  real swing_reach = 0;
  bool in_swing = false;
  long tick = 0;
  const double start = data->hands.front().capture_time + 0.5;
  const double end = data->hands.back().detect_time;

  for (double t = start; t < end; t += 0.001, ++tick) {
    while (frame < data->hands.size() && data->hands[frame].detect_time <= t) {
      const HandObservation& obs = data->hands[frame++];
      control.observe(obs);
      const Palm palm = palm_from(obs);
      if (palm.valid) {
        // The same mapping the filter applies, with no filtering and no delay.
        const PalmFilter& f = control.filter();
        const float u = std::clamp(0.5f + (palm.x - f.centre_x()) / f.span_x(), 0.0f, 1.0f);
        truth.push_back({obs.capture_time, static_cast<double>((2.0f * u - 1.0f) * kArmReach)});
      }
    }

    // ---- the ball, as recorded ----
    BallCue cue;
    const ChanceRecord* c = ball < balls.size() ? balls[ball] : nullptr;
    if (c != nullptr) {
      const real due = static_cast<real>(c->time - t) * speed;  // simulation seconds
      const real flight = c->notice * speed;
      // (The game knows where the ball is going from the moment it is struck.)
      cue = {.incoming = due < flight * 0.95f && due > -0.22f, .meeting_x = c->meeting.x, .due_in = due};
    }
    const HandControlOutput out = control.update(t, 0.001f, 0.001f * speed, cue);
    paddle.place(out.bat, {});
    if (cue.incoming) {
      paddle.set_meeting_point(c->meeting, cue.due_in, true);
    } else {
      paddle.clear_meeting_point();
    }
    paddle.advance(0.001f * speed, 0.001f);
    const Vec3 shown = paddle.pos() + out.display_offset;
    const Vec3 arm = shown - Vec3{out.body_x, 0, 0};  // the bat as the hand moves it

    // ---- swings: how far and how fast the drawn bat travels ----
    if (out.swing_began) {
      in_swing = true;
      swing_from = arm;
      swing_reach = 0;
    }
    if (in_swing) {
      swing_reach = std::max(swing_reach, length(arm - swing_from));
      if (!control.swing().live() && !out.holding) {
        in_swing = false;
        swing_travel.push_back(100.0 * static_cast<double>(swing_reach));
      }
    }

    // ---- would this ball be returned? ----
    if (c != nullptr && cue.incoming) {
      const real ball_z = paddle.pos().z - cue.due_in * c->ball_speed * 0.9f;
      const Vec3 gap = Vec3{c->meeting.x, c->meeting.y, ball_z} - paddle.pos();
      const bool in_reach = gap.x * gap.x + gap.y * gap.y <= kSwingReach * kSwingReach &&
                            gap.z >= -kSwingReachFront && gap.z <= kSwingReachBehind;
      if (control.swing().live() && gap.z >= -kSwingReachFront && gap.z <= kSwingReachBehind) {
        swung_near = true;  // a swing while the ball was passing: was the bat near enough?
        if (std::hypot(gap.x, gap.y) < std::hypot(nearest_side, nearest_height)) {
          nearest_side = std::abs(gap.x);
          nearest_height = std::abs(gap.y);
        }
      }
      if (!ball_hit && control.swing().live() && in_reach) {
        ball_hit = true;
        control.swing().consume();
        // Speed of the bat on screen at that moment, over the last 30 ms.
        double before = 0;
        if (at(drawn, t - 0.030, before)) {
          contact_speed.push_back(std::abs(static_cast<double>(arm.x) - before) / 0.030);
        }
      }
    }
    if (c != nullptr && t >= c->time + 0.30) {  // this ball is over
      ++counted;
      returned += ball_hit;
      if (!ball_hit) {
        if (!swung_near) ++missed_no_swing;
        else if (nearest_side > nearest_height) ++missed_sideways;
        else ++missed_height;
      }
      ball_hit = false;
      swung_near = false;
      nearest_side = nearest_height = 1e9f;
      ++ball;
    } else if (c != nullptr && std::abs(t - c->time) < 0.0005) {  // the moment it arrives
      moved_by_game.push_back(100.0 * static_cast<double>(length(paddle.pos() - paddle.raw_pos())));
      body_gap.push_back(100.0 * static_cast<double>(std::abs(out.body_x - c->meeting.x)));
      anticipated += std::abs(out.body_x - c->meeting.x) < 0.25f;
    }

    // ---- what the eye sees, 120 times a second ----
    if (tick % 8 != 0) continue;
    drawn.push_back({t, static_cast<double>(arm.x)});
    if (!out.hand_seen) {
      have_prev = false;
      continue;
    }
    const Vec3 step = have_prev ? arm - prev_arm : Vec3{};
    const bool calm = length(out.swing_velocity) < 0.25f && !out.holding;
    if (have_prev && calm) shake.push_back(1000.0 * static_cast<double>(length(step - prev_step)));
    prev_step = step;
    prev_arm = arm;
    have_prev = true;
  }

  // ---- the delay from hand to bat ----
  // The shift in time that makes the drawn bat best match where the hand
  // really was. Judged only while the hand is moving at a moderate pace (a
  // still hand matches at any shift, and a swing is held on purpose).
  double best_shift = 0, best_error = 1e300;
  for (double shift = 0.0; shift <= 0.300; shift += 0.004) {
    double error = 0;
    int n = 0;
    for (std::size_t i = 4; i + 4 < drawn.size(); i += 2) {
      double now_truth = 0, then_truth = 0, later_truth = 0;
      if (!at(truth, drawn[i].time - shift, now_truth)) continue;
      if (!at(truth, drawn[i].time - shift - 0.05, then_truth)) continue;
      if (!at(truth, drawn[i].time - shift + 0.05, later_truth)) continue;
      const double hand_speed = std::abs(later_truth - then_truth) / 0.1;  // m/s of arm travel
      if (hand_speed < 0.25 || hand_speed > 1.5) continue;
      const double d = drawn[i].value - now_truth;
      error += d * d;
      ++n;
    }
    if (n > 50 && error / n < best_error) {
      best_error = error / n;
      best_shift = shift;
    }
  }

  std::printf("Directness\n");
  std::printf("  hand to bat delay (excluding the display): %.0f ms    [target: 100 ms or less]\n",
              1000.0 * best_shift);
  std::printf("  bat shake with a calm hand, per displayed frame: median %.2f mm, 90%% %.2f mm, 99%% %.2f mm"
              "    [target: 90%% under 3 mm]\n",
              percentile(shake, 0.5), percentile(shake, 0.9), percentile(shake, 0.99));
  std::printf("\nWho moves the bat\n");
  std::printf("  moved by the magnet when the ball arrives: median %.0f cm, 90%% %.0f cm"
              "    [target: median under 10 cm]\n",
              percentile(moved_by_game, 0.5), percentile(moved_by_game, 0.9));
  std::printf("  body short of the ball when it arrives: median %.0f cm, 90%% %.0f cm; within 25 cm for %d of %zu\n",
              percentile(body_gap, 0.5), percentile(body_gap, 0.9), anticipated, body_gap.size());
  std::printf("\nThe swing on screen\n");
  std::printf("  bat travel during a swing: median %.0f cm, 90%% %.0f cm (%zu swings)\n",
              percentile(swing_travel, 0.5), percentile(swing_travel, 0.9), swing_travel.size());
  std::printf("  bat speed on screen at contact: median %.1f m/s, 25%% %.1f m/s    [target: above 1 m/s]\n",
              percentile(contact_speed, 0.5), percentile(contact_speed, 0.25));
  std::printf("\nReturning\n");
  int played = 0;
  for (const ChanceRecord* c : balls) played += c->hit;
  std::printf("  as played: %d of %zu returned; replayed with this build: %d of %d (the ball is not"
              " re-simulated, so this is a guide)\n",
              played, balls.size(), returned, counted);
  std::printf("  missed in the replay: %d with no swing while the ball was passing, %d with the bat off to"
              " the side, %d with the bat too high or low\n",
              missed_no_swing, missed_sideways, missed_height);
  return 0;
}
