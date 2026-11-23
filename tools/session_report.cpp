// Turns a recorded play session into the numbers used to tune the game.
//
//   session_report            newest session
//   session_report <dir>      a specific session directory
//
// Every section says which setting its numbers are for.
#include "estimation/palm.hpp"
#include "estimation/palm_filter.hpp"
#include "game/player_paddle.hpp"
#include "game/session_log.hpp"
#include "game/swing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace pp;

namespace {

// p in [0, 1]. Sorts its argument.
double percentile(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())))];
}

double mean(const std::vector<double>& v) {
  double sum = 0;
  for (const double x : v) sum += x;
  return v.empty() ? 0 : sum / static_cast<double>(v.size());
}

void report_chances(const std::vector<ChanceRecord>& all, bool hand) {
  std::vector<const ChanceRecord*> chances;
  for (const ChanceRecord& c : all) {
    if (c.hand == hand) chances.push_back(&c);
  }
  std::printf("\n== Returning the ball, %s (%zu chances) ==\n", hand ? "hand" : "mouse",
              chances.size());
  if (chances.size() < 5) {
    std::printf("  too few to say anything\n");
    return;
  }

  const SwingConfig swing_cfg = hand ? kHandSwing : kMouseSwing;
  int hits = 0, never_settled = 0;
  std::vector<double> miss, settle, swing, notice, dx, dy;
  std::array<int, 5> results{};
  std::array<std::array<int, 2>, 6> zones{};  // [left|centre|right] x [low|high] -> {chances, hits}
  for (const ChanceRecord* c : chances) {
    hits += c->hit;
    miss.push_back(static_cast<double>(c->miss_raw));
    dx.push_back(static_cast<double>(c->raw.x - c->meeting.x));
    dy.push_back(static_cast<double>(c->raw.y - c->meeting.y));
    notice.push_back(static_cast<double>(c->notice));
    if (c->settled_for >= 0) settle.push_back(static_cast<double>(c->settled_for));
    else ++never_settled;
    if (c->hit) {
      swing.push_back(static_cast<double>(length(c->paddle_vel)));
      ++results[static_cast<std::size_t>(c->result)];
    }
    const std::size_t column = c->meeting.x < -0.2f ? 0 : c->meeting.x > 0.2f ? 2 : 1;
    const std::size_t zone = column + (c->meeting.y > 0.92f ? 3 : 0);
    ++zones[zone][0];
    zones[zone][1] += c->hit;
  }
  const double n = static_cast<double>(chances.size());

  std::printf("  returned %d of %zu (%.0f%%)\n", hits, chances.size(), 100.0 * hits / n);

  // --- magnet reach ---
  const MagnetConfig magnet = hand ? kHandMagnet : kMouseMagnet;
  int inside_full = 0, inside_any = 0;
  for (const double m : miss) {
    inside_full += m <= static_cast<double>(magnet.reach_full);
    inside_any += m <= static_cast<double>(magnet.reach_none);
  }
  std::printf("\n  Aim error before any help (paddle to ball, at arrival):\n");
  std::printf("    half within %.0f cm, 80%% within %.0f cm, 95%% within %.0f cm\n",
              100 * percentile(miss, 0.50), 100 * percentile(miss, 0.80),
              100 * percentile(miss, 0.95));
  std::printf("    magnet at full strength covers %.0f%%, any strength %.0f%%"
              "   [MagnetConfig reach_full %.2f, reach_none %.2f]\n",
              100.0 * inside_full / n, 100.0 * inside_any / n,
              static_cast<double>(magnet.reach_full), static_cast<double>(magnet.reach_none));

  // --- systematic offset ---
  std::printf("\n  Which way the paddle misses, on average:\n");
  std::printf("    %+.1f cm sideways (+ = too far right), %+.1f cm vertically (+ = too high)\n",
              100 * mean(dx), 100 * mean(dy));
  std::printf("    A steady offset is a mapping or prediction error, not the player's.\n");

  // --- why balls were missed ---
  int no_swing = 0, too_early = 0, too_late = 0, out_of_reach = 0;
  std::vector<double> lead_of_hits;
  for (const ChanceRecord* c : chances) {
    if (c->hit) {
      if (c->swing_lead != ChanceRecord::kNoSwing) lead_of_hits.push_back(static_cast<double>(c->swing_lead));
      continue;
    }
    if (c->swing_lead == ChanceRecord::kNoSwing) ++no_swing;
    else if (c->swing_lead > static_cast<real>(swing_cfg.live_for)) ++too_early;
    else if (c->swing_lead < -0.06f) ++too_late;
    else ++out_of_reach;
  }
  std::printf("\n  Missed balls: %d with no swing, %d swung too early, %d too late,"
              " %d swung in time but out of reach\n",
              no_swing, too_early, too_late, out_of_reach);
  if (lead_of_hits.size() >= 3) {
    std::printf("  Swings that hit began %.2f s before the ball arrived (median; range %.2f to %.2f)\n",
                percentile(lead_of_hits, 0.5), percentile(lead_of_hits, 0.0),
                percentile(lead_of_hits, 1.0));
    std::printf("    [SwingConfig live_for %.2f: a swing must begin no earlier than this]\n",
                static_cast<double>(swing_cfg.live_for));
  }

  // --- reaction ---
  std::printf("\n  Time:\n");
  std::printf("    ball in flight toward you for %.2f s (median)\n", percentile(notice, 0.5));
  std::printf("    paddle in place %.2f s before arrival (median); never in place %.0f%%\n",
              percentile(settle, 0.5), 100.0 * never_settled / n);

  // --- swing ---
  if (!swing.empty()) {
    std::printf("\n  Paddle speed when hitting: median %.1f m/s, 90%% below %.1f m/s, max %.1f\n",
                percentile(swing, 0.5), percentile(swing, 0.9), percentile(swing, 1.0));
    std::printf("  Shots: %d good, %d net, %d out, %d own side\n",
                results[static_cast<std::size_t>(ShotResult::Good)],
                results[static_cast<std::size_t>(ShotResult::Net)],
                results[static_cast<std::size_t>(ShotResult::Out)],
                results[static_cast<std::size_t>(ShotResult::OwnSide)]);
  }

  // --- where on the table ---
  std::printf("\n  Returned, by where the ball arrived (left / centre / right):\n");
  for (const std::size_t row : {std::size_t{3}, std::size_t{0}}) {
    std::printf("    %s", row == 3 ? "high" : "low ");
    for (std::size_t col = 0; col < 3; ++col) {
      const auto& z = zones[row + col];
      if (z[0] == 0) std::printf("      -     ");
      else std::printf("   %3d%% of %-3d", 100 * z[1] / z[0], z[0]);
    }
    std::printf("\n");
  }
}

void report_tracking(const std::vector<HandObservation>& hands) {
  std::printf("\n== Hand tracking (%zu camera frames) ==\n", hands.size());
  if (hands.size() < 30) {
    std::printf("  too few to say anything\n");
    return;
  }

  int found = 0, usable = 0;
  std::vector<double> latency, gaps, speeds, xs, ys, sizes, noise;
  std::vector<Palm> palms(hands.size());
  for (std::size_t i = 0; i < hands.size(); ++i) {
    const HandObservation& h = hands[i];
    palms[i] = palm_from(h);
    found += h.found;
    usable += palms[i].valid;
    latency.push_back(1000 * (h.detect_time - h.capture_time));
    if (i > 0) gaps.push_back(1000 * (h.capture_time - hands[i - 1].capture_time));
    if (palms[i].valid) {
      xs.push_back(static_cast<double>(palms[i].x));
      ys.push_back(static_cast<double>(palms[i].y));
      sizes.push_back(static_cast<double>(apparent_hand_size(h)));
    }
  }

  // Measurement noise, estimated from the data itself. Over three consecutive
  // frames a real hand moves almost in a straight line, so how far the middle
  // sample sits from the midpoint of its neighbours is nearly all noise.
  // For independent noise of deviation s, that residual has deviation
  // s * sqrt(1.5). Only slow stretches are used, where the straight-line
  // assumption is safest.
  for (std::size_t i = 1; i + 1 < hands.size(); ++i) {
    if (!palms[i - 1].valid || !palms[i].valid || !palms[i + 1].valid) continue;
    const double dt = hands[i + 1].capture_time - hands[i - 1].capture_time;
    if (dt <= 0 || dt > 0.12) continue;
    const double speed =
        std::hypot(static_cast<double>(palms[i + 1].x - palms[i - 1].x),
                   static_cast<double>(palms[i + 1].y - palms[i - 1].y)) / dt;
    speeds.push_back(speed);
    if (speed < 0.10) {
      const double residual = static_cast<double>(palms[i].x) -
                              0.5 * static_cast<double>(palms[i - 1].x + palms[i + 1].x);
      noise.push_back(residual * residual);
    }
  }

  const double n = static_cast<double>(hands.size());
  std::printf("  hand seen in %.0f%% of frames; usable palm in %.0f%%\n", 100.0 * found / n,
              100.0 * usable / n);
  std::printf("  camera delay (exposure to result): median %.0f ms, 99%% under %.0f ms\n",
              percentile(latency, 0.5), percentile(latency, 0.99));
  std::printf("  frame interval: median %.1f ms, 99%% under %.1f ms\n", percentile(gaps, 0.5),
              percentile(gaps, 0.99));

  const PalmFilterConfig cfg;
  if (noise.size() >= 20) {
    const double sigma = std::sqrt(mean(noise) / 1.5);
    std::printf("\n  Measurement noise of a slow hand: %.4f of image width (%.1f px at 640)\n",
                sigma, sigma * 640);
    std::printf("    [PalmFilterConfig pos_noise is %.4f: it should match this]\n",
                static_cast<double>(cfg.pos_noise));
  }
  if (!speeds.empty()) {
    std::printf("  Hand speed: median %.2f, 95%% below %.2f, max %.2f image widths per second\n",
                percentile(speeds, 0.5), percentile(speeds, 0.95), percentile(speeds, 1.0));
  }
  if (!xs.empty()) {
    std::printf("\n  Where the hand was (image fraction, 5th to 95th percentile):\n");
    const double x_lo = percentile(xs, 0.05), x_hi = percentile(xs, 0.95);
    const double y_lo = percentile(ys, 0.05), y_hi = percentile(ys, 0.95);
    std::printf("    x %.2f to %.2f (travel %.2f),  y %.2f to %.2f (travel %.2f)\n", x_lo, x_hi,
                x_hi - x_lo, y_lo, y_hi, y_hi - y_lo);
    std::printf("    [PalmFilterConfig span_x %.2f, span_y %.2f: travel that covers the paddle's"
                " whole range]\n",
                static_cast<double>(cfg.span_x), static_cast<double>(cfg.span_y));
    const double near = percentile(sizes, 0.95), far = percentile(sizes, 0.05);
    std::printf("    apparent size %.3f to %.3f: a reach range of about %.0f cm of paddle travel"
                "   [depth_gain %.1f]\n",
                far, near, 100 * static_cast<double>(cfg.depth_gain) * std::log(near / far),
                static_cast<double>(cfg.depth_gain));
  }
}

void report_swings(const std::vector<SwingRecord>& all, bool hand) {
  std::vector<double> peak, hit_peak, due;
  int hits = 0, at_nothing = 0;
  for (const SwingRecord& s : all) {
    if (s.hand != hand) continue;
    peak.push_back(static_cast<double>(s.peak_speed));
    hits += s.hit;
    if (s.hit) hit_peak.push_back(static_cast<double>(s.peak_speed));
    if (s.ball_due_in < 0) ++at_nothing;
    else due.push_back(static_cast<double>(s.ball_due_in));
  }
  std::printf("\n== Swings, %s (%zu) ==\n", hand ? "hand" : "mouse", peak.size());
  if (peak.size() < 5) {
    std::printf("  too few to say anything\n");
    return;
  }
  const SwingConfig cfg = hand ? kHandSwing : kMouseSwing;
  const char* unit = hand ? "image widths/s" : "m/s";
  std::printf("  %d hit the ball; %d were made with no ball on its way (possible false triggers)\n",
              hits, at_nothing);
  std::printf("  peak speed: 10%% below %.2f, median %.2f, 90%% below %.2f %s\n",
              percentile(peak, 0.1), percentile(peak, 0.5), percentile(peak, 0.9), unit);
  std::printf("    [SwingConfig start_speed %.2f, full_speed %.2f: a swing begins at the first,"
              " full power at the second]\n",
              static_cast<double>(cfg.start_speed), static_cast<double>(cfg.full_speed));
  if (!due.empty()) {
    std::printf("  when a ball was coming, swings began %.2f s before it was due (median)\n",
                percentile(due, 0.5));
  }
}

void report_points(const std::vector<PointRecord>& points) {
  std::printf("\n== Points (%zu) ==\n", points.size());
  if (points.empty()) return;
  int won = 0;
  double rally = 0;
  std::array<std::array<int, 2>, 9> reasons{};  // [reason][winner]
  for (const PointRecord& p : points) {
    won += p.winner == Side::Player;
    rally += p.rally_hits;
    ++reasons[static_cast<std::size_t>(p.reason)][index(p.winner)];
  }
  std::printf("  you won %d, opponent %zu; average rally %.1f strokes\n", won,
              points.size() - static_cast<std::size_t>(won),
              rally / static_cast<double>(points.size()));
  std::printf("  how points ended                you won   you lost\n");
  for (std::size_t r = 0; r < reasons.size(); ++r) {
    if (reasons[r][0] + reasons[r][1] == 0) continue;
    std::printf("    %-28s %6d %10d\n", describe(static_cast<PointReason>(r)), reasons[r][0],
                reasons[r][1]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path dir;
  if (argc > 1) {
    dir = argv[1];
  } else if (const auto newest = latest_session(SessionLog::default_root())) {
    dir = *newest;
  } else {
    std::printf("no sessions under %s\n", SessionLog::default_root().c_str());
    return 1;
  }

  const auto data = load_session(dir);
  if (!data) {
    std::printf("cannot read a session from %s\n", dir.c_str());
    return 1;
  }
  std::printf("session %s\n", dir.c_str());
  report_points(data->points);
  report_chances(data->chances, /*hand=*/true);
  report_chances(data->chances, /*hand=*/false);
  report_swings(data->swings, /*hand=*/true);
  report_swings(data->swings, /*hand=*/false);
  report_tracking(data->hands);
  return 0;
}
