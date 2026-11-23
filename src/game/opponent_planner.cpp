#include "game/opponent_planner.hpp"

#include "game/shot.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace pp {

namespace {

real unit(std::uint32_t& state) {  // xorshift32, as in match.cpp
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<real>(state >> 8) * (1.0f / 16777216.0f);
}

real lerp1(real a, real b, int i, int n) {
  return a + (b - a) * static_cast<real>(i) / static_cast<real>(n - 1);
}

}  // namespace

OpponentPlanner::OpponentPlanner() { batch_.resize(kCandidates); }

OpponentPlanner& opponent_planner() {
  thread_local OpponentPlanner planner;
  return planner;
}

std::optional<OpponentShot> OpponentPlanner::choose(const World& world,
                                                    const OpponentSkill& skill,
                                                    real player_bat_x, std::uint32_t& rng) {
  const Table& t = world.table;
  const real half_w = t.width * 0.5f, half_l = t.length * 0.5f;
  const Vec3 from = world.ball.pos;

  // ---- lay out the candidates ----
  // Directions fan from one far corner of the player's half to the other.
  const real left = std::atan2(-half_w - from.x, half_l * 0.6f - from.z);
  const real right = std::atan2(half_w - from.x, half_l * 0.6f - from.z);
  const real spins[kSpins] = {20.0f, 55.0f};  // topspin for travel toward +z: rotation about +x
  std::size_t n = 0;
  for (int h = 0; h < kHeadings; ++h) {
    const real heading = lerp1(left, right, h, kHeadings);
    for (int s = 0; s < kSpeeds; ++s) {
      const real speed = lerp1(skill.pace_lo, skill.pace_hi, s, kSpeeds);
      for (int l = 0; l < kLifts; ++l) {
        const real lift = lerp1(-0.6f, 3.0f, l, kLifts);
        for (int k = 0; k < kSpins; ++k) {
          batch_.set(n++, {.vel = {speed * std::sin(heading), lift, speed * std::cos(heading)},
                           .spin = {spins[k], 0, 0}});
        }
      }
    }
  }

  // ---- fly them all ----
  // In 8 ms steps, four times coarser than the batch simulator's default.
  // That looks reckless and is not: the integrator is fourth-order, so its
  // error shrinks with the fourth power of the step, and a ball in free
  // flight is a smooth problem. Measured against the full 1 ms simulation
  // over the playable candidates, landing points differ by at most 0.2 mm
  // at 8 ms (0.0 mm at 2 ms). The cost, though, is proportional to the
  // number of steps: 619 us per decision at 2 ms, 153 us at 8 ms.
  const ShotQuery query{.from = from,
                        .ball = world.ball_params,
                        .land_y = t.height + world.ball_params.radius,
                        .net_top = t.height + world.net.height,
                        .dt = 8e-3f,
                        .max_steps = 175};
  simulate_neon(query, batch_, 0, batch_.padded());

  // ---- score the playable ones ----
  // A weaker player leaves more room for error; a stronger one plays closer
  // to the lines and lower over the net.
  const real line_margin = 0.06f + 0.07f * skill.errors;
  const real net_margin = 0.025f + 0.030f * skill.errors;
  // What a shot is worth:
  //   away    how far it lands from the player's bat, counted only up to
  //           half the table's width. Beyond that the player is beaten
  //           anyway; rewarding still more distance is what sent 52 of 58
  //           balls to the side lines in recorded play.
  //   quick   how little time it gives
  //   deep    how far back it lands
  //   safe    how far inside the side lines: the percentage shot, which is
  //           what players hit most of the time
  // and then a draw. Real players do not pick the single best shot every
  // time, and an opponent who did would be both unbeatable in one way and
  // perfectly predictable in another. The draw is Gumbel noise, which makes
  // "take the highest total" the same as choosing each shot with probability
  // proportional to exp(score / spread): good shots often, the best most
  // often, anything playable sometimes.
  const real care = 0.3f + 1.3f * skill.placement;
  const real caution = 1.1f - 0.9f * skill.placement;
  const real spread = 0.45f;

  std::array<real, kCandidates> score;
  int playable = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(kCandidates); ++i) {
    const ShotOutcome o = batch_.outcome(i);
    const bool ok = o.time > 0 && o.clearance >= net_margin &&
                    std::abs(o.land_x) <= half_w - line_margin && o.land_z >= 0.30f &&
                    o.land_z <= half_l - line_margin;
    if (!ok) {
      score[i] = -1e9f;
      continue;
    }
    ++playable;
    const real away = std::min(std::abs(o.land_x - player_bat_x) / t.width, 0.5f) * 2.0f;  // 0..1
    const real quick = std::clamp(1.0f - o.time / 0.8f, 0.0f, 1.0f);
    const real deep = o.land_z / half_l;
    const real wide = std::abs(o.land_x) / half_w;  // 0 middle .. 1 side line
    const real gumbel = -std::log(-std::log(std::clamp(unit(rng), 1e-6f, 1.0f - 1e-6f)));
    score[i] = care * away + 1.0f * quick + 0.6f * deep - caution * wide * wide + spread * gumbel;
  }
  if (playable == 0) return std::nullopt;

  // ---- take the best that really works ----
  // The batch flies the ball through the air only. The few leading
  // candidates are checked with the full simulation (bounce included): a
  // ball that lands but then dies on the table could not be played back.
  for (int attempt = 0; attempt < 8; ++attempt) {
    const auto best = static_cast<std::size_t>(
        std::max_element(score.begin(), score.end()) - score.begin());
    if (score[best] < -1e8f) break;
    const Vec3 vel{batch_.vx[best], batch_.vy[best], batch_.vz[best]};
    const Vec3 spin{batch_.wx[best], batch_.wy[best], batch_.wz[best]};
    if (stroke_carries(world, {.pos = from, .vel = vel, .spin = spin})) {
      const ShotOutcome o = batch_.outcome(best);
      return OpponentShot{vel, spin, {o.land_x, t.height, o.land_z}, kCandidates, playable};
    }
    score[best] = -1e9f;
  }
  return std::nullopt;
}

}  // namespace pp
