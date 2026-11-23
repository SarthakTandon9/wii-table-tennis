#pragma once
// One game of table tennis, by the rules.
//
// Rules implemented (ITTF laws, simplified where noted):
//  - Game to 11, win by 2. Serve changes every 2 points, every point from 10-10.
//  - Serve: must bounce once on the server's half, then on the receiver's.
//    Touching the net on an otherwise good serve is a let (replayed).
//  - Rally: the ball must bounce exactly once on your half before you hit it,
//    and your return must land on the other half without touching yours.
//  - Volley: touching the ball before it bounces loses the point if it was
//    still over the table. If it had already flown past the table it was
//    going out, so the striker loses instead.
//  - Double bounce, into the net, off the table: point to the other side.
// Simplified: the player's toss is automatic and a missed toss is simply
// re-tossed (strictly it is a fault). The opponent serves without a toss.

#include "core/types.hpp"
#include "math/vec3.hpp"
#include "physics/world.hpp"

#include <array>
#include <cstdint>

namespace pp {

enum class Side : std::uint8_t { Player = 0, Opponent = 1 };
constexpr Side other(Side s) { return s == Side::Player ? Side::Opponent : Side::Player; }
constexpr std::size_t index(Side s) { return static_cast<std::size_t>(s); }

enum class Phase : std::uint8_t {
  PreServe,   // ball held, short pause before the serve
  Rally,      // ball in play (includes the toss)
  PointOver,  // point decided, ball left to bounce out
  GameOver,
};

// What must happen next for the rally to stay alive.
enum class Stage : std::uint8_t {
  ServeToss,       // tossed, waiting for the server to strike it
  ServeOwnBounce,  // struck, must bounce on the server's half
  ServeFarBounce,  // then on the receiver's half
  AwaitBounce,     // a stroke is in flight, must bounce on the other half
  AwaitHit,        // it bounced; that side must hit it before it bounces again
};

enum class PointReason : std::uint8_t {
  None,
  Let,
  ServeFault,
  Net,
  Out,
  OwnSide,
  DoubleBounce,
  Missed,
  Volley,
};
const char* describe(PointReason r);

// The kind of stroke a swing was read as, from the direction the hand moved.
// A hit does not have to be a forward movement: brushing upward and cutting
// downward are strokes too, and each sends a different ball.
enum class StrokeKind : std::uint8_t {
  Drive,    // forward or across: flat and fast
  Topspin,  // upward: arcs higher, dips, kicks forward off the bounce
  Chop,     // downward: slower, backspin, stays low
};

// How well the swing was timed against the ball.
enum class StrokeTiming : std::uint8_t {
  Good,
  Early,  // the swing began long before the ball arrived
  Late,   // the ball had already gone by the bat when the swing caught it
};

// Where the player's paddle is right now. Produced by the mouse today and by
// the hand tracker + Kalman filter from M4 on.
struct PaddleInput {
  Vec3 pos;
  Vec3 vel;
  Vec3 normal{0, 0, -1};

  // Swing play (see game/swing.hpp). With `swing_mode` set, having the bat in
  // the ball's path does nothing by itself: the ball is struck only while
  // `swinging`, when it comes within reach of the bat. How hard and which way
  // the hand was moving decide the shot.
  // With `swing_mode` clear, the bat is a plain physical paddle: touching the
  // ball returns it. (The scripted players in the tests use that.)
  bool swing_mode = false;
  bool swinging = false;
  real swing_power = 0;  // 0 = the gentlest movement that counts, 1 = a full swing
  Vec3 swing_dir;        // unit vector: +x to the right, +y upward, +z a push forward
  real swing_age = 0;    // s of real time since the swing began
  // The attitude of the bat's face, from the hand (see game/bat_face.hpp).
  // Both zero when it is not known (mouse). The face is drawn, and with
  // aim assist off it is the surface the ball bounces from; with assist on
  // it does not steer the shot (see assist_swing_stroke for why not).
  real face_turn = 0;    // radians, + = facing right: the ball goes that way
  real face_closed = 0;  // radians, tipped forward over the ball
};

// How good the opponent is. Three things, the same three that separate real
// players: how quickly it gets to the ball, how soon it reads the shot, and
// how often it makes a mistake. (Pace and placement of its own shots too.)
struct OpponentSkill {
  real speed;      // m/s its bat can travel
  real reaction;   // s before it reads a new shot
  real errors;     // multiplies its chance of an outright error
  real pace_lo, pace_hi;  // m/s: the range its own shots are hit at
  real width;      // 0..1: how close to the side lines it aims
  real placement;  // 0..1: how often it deliberately plays away from the player's bat
};
inline constexpr int kDifficultyLevels = 5;
inline constexpr int kDefaultDifficulty = 2;
inline constexpr OpponentSkill kOpponentSkill[kDifficultyLevels] = {
    {2.3f, 0.22f, 1.50f, 5.2f, 6.6f, 0.40f, 0.05f},  // beginner
    {2.7f, 0.19f, 1.20f, 5.6f, 7.2f, 0.48f, 0.15f},  // casual
    {2.9f, 0.17f, 1.00f, 6.0f, 8.0f, 0.55f, 0.30f},  // club
    {3.3f, 0.15f, 0.80f, 6.6f, 8.8f, 0.64f, 0.45f},  // strong
    {3.7f, 0.13f, 0.60f, 7.2f, 9.6f, 0.72f, 0.60f},  // expert
};
const char* difficulty_name(int level);

struct Match {
  World world;
  std::array<Paddle, 2> paddles;  // indexed by Side

  std::array<int, 2> score{};
  Side first_server = Side::Player;
  Side server = Side::Player;
  Phase phase = Phase::PreServe;
  Stage stage = Stage::ServeToss;
  Side striker = Side::Player;  // who hit the ball last (the server, before the serve)

  PointReason last_reason = PointReason::None;
  Side last_winner = Side::Player;
  int rally_hits = 0;  // strokes in the current rally, both sides

  // A swing that has connected but not yet sent the ball away (see kStrikeHold).
  real strike_hold = 0;
  real strike_incoming_speed = 0;
  real strike_swing_age = 0;  // how long the swing had been going when it connected
  real strike_gap_z = 0;      // how far past the bat the ball was at that moment (+ = behind)

  PaddleInput input;          // the player's input for the current tick
  int player_hits = 0;        // legal strokes by the player since the match began
  real last_swing_age = 0;    // swing_age at the player's most recent stroke
  StrokeKind last_stroke = StrokeKind::Drive;  // what that stroke was read as
  StrokeTiming last_timing = StrokeTiming::Good;
  Vec3 last_target;           // where on the far half that stroke was sent

  // Aim assist for the player: the stroke keeps its direction and pace but is
  // re-lofted so that it clears the net and lands on the table. With a mouse
  // there is no way to angle the paddle, so without this most hits sail long.
  bool assist = true;
  int difficulty = kDefaultDifficulty;  // index into kOpponentSkill
  real last_hit_speed = 0;  // m/s the player's latest stroke left the bat at

  // Internals.
  real timer = 1.0f;                   // countdown in PreServe / PointOver
  real quiet = 0;                      // seconds since anything touched the ball
  bool serve_touched_net = false;
  std::array<real, 2> hit_cooldown{};  // suppresses double contacts
  Vec3 ai_goal;
  real ai_wait = 0;                    // reaction delay before the AI replans
  bool ai_replan = false;
  std::uint32_t rng;

  explicit Match(Side first = Side::Player, std::uint32_t seed = 0x9E3779B9u);
};

// Scoring helpers, exposed for tests.
Side server_for(int points_played, Side first_server);
bool game_won(int winner_score, int loser_score);

// One fixed simulation tick. Returns what the ball touched.
ContactMask tick(Match& m, const PaddleInput& input, real dt);

// In swing play, how close the ball must be to the bat to be struck: within
// kSwingReach sideways, and between kSwingReachFront ahead of the bat and
// kSwingReachBehind past it. Generous on purpose, and much deeper behind than
// in front. A swing is seen late (the camera's delay plus the time it takes
// for a movement to register as fast), so by the time the game knows, the
// ball has often gone by: one recorded swing was 0.14 s late, which at this
// ball speed is 0.5 m. Motion games accept that; so does this one.
inline constexpr real kSwingReach = 0.30f;
inline constexpr real kSwingReachFront = 0.28f;
inline constexpr real kSwingReachBehind = 0.55f;

// When a swing connects, the ball is held on the bat for this long (simulation
// seconds) before it is sent away.
//
// The reason is measurement. A swing is recognised at the moment the hand's
// speed first crosses the threshold, so at that instant every swing looks
// like the gentlest possible one: in recorded play every hit scored a power
// of 0.03 to 0.12 however hard the hand was thrown. Holding the ball for a
// few camera frames lets the swing reach its real peak before the shot is
// decided. It also reads as impact, which games add on purpose ("hit stop").
inline constexpr real kStrikeHold = 0.045f;

// Where the player's ball rests before the toss, given the paddle position.
Vec3 toss_point(const Match& m, Vec3 paddle_pos);

}  // namespace pp
