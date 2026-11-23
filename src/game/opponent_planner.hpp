#pragma once
// How the opponent chooses its shot.
//
// The first opponent picked a random spot, asked a solver for the one
// velocity that lands there, and hit that. It never weighed one shot against
// another, so it never played the ball that was hard for the player in
// particular.
//
// This one does what a player does: it considers many shots and picks the
// best one it can make. Several hundred candidate velocities (directions
// across the whole far half, a range of pace, a range of lift, two amounts of
// topspin) are flown forward through the air at once by the batch simulator
// (src/planner/shot_batch: four balls per NEON register). Those that clear
// the net and land on the table with room to spare are scored:
//
//   - how far the ball lands from where the player's bat is (make them move)
//   - how fast it gets there (give them less time)
//   - how deep it lands (push them back)
//
// and the best is played, after a full simulation with the bounce confirms
// that the ball will carry and can be played back. Skill decides how much
// margin from the lines and the net the opponent insists on, how much it
// cares about placement, and how consistently it finds the best option.
//
// Cost: 576 flights per decision in about 0.2 ms on one core, once per
// opponent stroke, inside the simulation thread's 1 ms tick. The
// one-at-a-time solver takes 0.08 ms for a SINGLE target; weighing even a
// few dozen alternatives that way would not fit in a tick.

#include "core/types.hpp"
#include "game/match.hpp"
#include "math/vec3.hpp"
#include "physics/world.hpp"
#include "planner/shot_batch.hpp"

#include <cstdint>
#include <optional>

namespace pp {

struct OpponentShot {
  Vec3 vel, spin;
  Vec3 landing;         // where the planner expects it to bounce
  int considered = 0;   // candidates simulated
  int playable = 0;     // of those, how many were legal with the margins asked for
};

class OpponentPlanner {
 public:
  OpponentPlanner();  // allocates the batch; nothing allocates after this

  // The shot to play from `world.ball.pos` toward the player's half (+z), or
  // nothing if no candidate is playable (the caller then falls back to the
  // single-shot solver). `player_bat_x` is where the player's bat is now.
  std::optional<OpponentShot> choose(const World& world, const OpponentSkill& skill,
                                     real player_bat_x, std::uint32_t& rng);

  static constexpr int kHeadings = 8, kSpeeds = 6, kLifts = 6, kSpins = 2;
  static constexpr int kCandidates = kHeadings * kSpeeds * kLifts * kSpins;

 private:
  ShotBatch batch_;
};

// One planner per thread, created on first use. Call this once up front on a
// thread that must not allocate later (the simulation thread does).
OpponentPlanner& opponent_planner();

}  // namespace pp
