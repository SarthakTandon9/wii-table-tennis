#include "game_app.hpp"

#include "audio/sounds.hpp"
#include "core/signpost.hpp"
#include "game/paddle_surface.hpp"
#include "game/session_log.hpp"
#include "game/sim_thread.hpp"
#include "platform/hand_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>

namespace pp {

namespace {

using render::Color;
using render::Mat4;
using render::Material;
using render::Mesh;
using render::MetalRenderer;
using render::rgb;

constexpr float kPi = std::numbers::pi_v<float>;

constexpr Color kAmber = rgb(250, 176, 60);
constexpr Color kGreen = rgb(96, 196, 150);
constexpr Color kRed = rgb(214, 92, 80);
constexpr Color kWhite = rgb(240, 240, 236);
constexpr Color kLight = rgb(200, 200, 200);
constexpr Color kGray = rgb(130, 130, 130);

// printf into a fixed buffer: text for the overlay without touching the heap.
template <typename... Args>
std::string_view fmt(std::array<char, 160>& buffer, const char* format, Args... args) {
  const int n = std::snprintf(buffer.data(), buffer.size(), format, args...);
  return {buffer.data(), static_cast<std::size_t>(std::clamp(n, 0, static_cast<int>(buffer.size()) - 1))};
}

// ----------------------------------------------------------------- the hall

void draw_hall(MetalRenderer& r) {
  // Sports flooring, and the low boards that ring a competition court.
  r.draw(Mesh::Box, render::translation({0, -0.01f, 0}) * render::scaling({18, 0.02f, 18}),
         rgb(112, 40, 38), Material::Floor, false);
  const Color board = rgb(20, 48, 92);
  for (const float x : {-3.3f, 3.3f}) {
    r.draw(Mesh::Box, render::translation({x, 0.38f, -1.0f}) * render::scaling({0.04f, 0.76f, 9.0f}),
           board, Material::Matte);
  }
  r.draw(Mesh::Box, render::translation({0, 0.38f, -5.5f}) * render::scaling({6.64f, 0.76f, 0.04f}),
         board, Material::Matte);
  // The far wall, mostly lost in the dark.
  r.draw(Mesh::Box, render::translation({0, 3.0f, -9.0f}) * render::scaling({24, 6, 0.1f}),
         rgb(58, 64, 78), Material::Matte, false);
}

void draw_table(MetalRenderer& r, const Table& t, const Net& net) {
  r.draw(Mesh::Box,
         render::translation({0, t.height - 0.011f, 0}) * render::scaling({t.width, 0.022f, t.length}),
         rgb(20, 62, 112), Material::Table);
  // Frame under the top, and four legs.
  const Color steel = rgb(38, 40, 46);
  r.draw(Mesh::Box,
         render::translation({0, t.height - 0.045f, 0}) *
             render::scaling({t.width - 0.10f, 0.045f, t.length - 0.16f}),
         steel, Material::Gloss);
  for (const float x : {-0.58f, 0.58f}) {
    for (const float z : {-1.05f, 1.05f}) {
      r.draw(Mesh::Cylinder, render::along({x, 0, z}, {x, t.height - 0.06f, z}, 0.022f), steel,
             Material::Gloss);
    }
  }
  // Net and its posts.
  const float post_x = t.width * 0.5f + net.overhang;
  for (const float x : {-post_x, post_x}) {
    r.draw(Mesh::Cylinder,
           render::along({x, t.height - 0.03f, 0}, {x, t.height + net.height + 0.004f, 0}, 0.007f),
           steel, Material::Gloss);
  }
  r.draw(Mesh::Box,
         render::translation({0, t.height + net.height * 0.5f, 0}) *
             render::scaling({2 * post_x, net.height, 0.003f}),
         rgb(30, 30, 30), Material::Net);
}

// A bat: wooden blade, rubber on both faces, and a handle. The handle hangs
// down and leans toward the body, so it swaps sides as the bat crosses the
// middle, the way a forehand turns into a backhand. With a tracked hand,
// `roll` turns the bat the way the hand is turned.
// Returns the end of the handle (where a hand would hold it).
Vec3 draw_bat(MetalRenderer& r, const Paddle& p, Color front, Color back, float alpha,
              bool use_roll = false, real roll = 0) {
  const Color wood = rgb(196, 154, 98, alpha);
  const Vec3 n = normalized(p.normal);
  Vec3 down = use_roll ? Vec3{-std::sin(roll), -std::cos(roll), 0}
                       : Vec3{std::clamp(-p.pos.x * 1.2f, -0.6f, 0.6f), -1, 0};
  down = normalized(down - dot(down, n) * n);  // lie in the blade's plane

  const Vec3 grip = p.pos + down * (p.radius + 0.105f);
  r.draw(Mesh::Cylinder, render::along(p.pos + down * (p.radius * 0.85f), grip, 0.014f), wood,
         Material::Gloss);
  r.draw(Mesh::Cylinder, render::along(p.pos - n * 0.004f, p.pos + n * 0.004f, p.radius), wood,
         Material::Gloss);
  r.draw(Mesh::Cylinder,
         render::along(p.pos + n * 0.004f, p.pos + n * 0.007f, p.radius * 0.97f),
         render::fade(front, alpha), Material::Rubber);
  r.draw(Mesh::Cylinder,
         render::along(p.pos - n * 0.007f, p.pos - n * 0.004f, p.radius * 0.97f),
         render::fade(back, alpha), Material::Rubber);
  return grip;
}

// A two-bone arm from `shoulder` to `hand`: finds where the elbow must be for
// an upper arm and forearm of the given lengths, bending downward.
void draw_arm(MetalRenderer& r, Vec3 shoulder, Vec3 hand, float upper, float fore, float radius,
              Color sleeve, Color skin) {
  const Vec3 reach = hand - shoulder;
  const float d = std::clamp(length(reach), 0.05f, upper + fore - 0.005f);
  const Vec3 dir = normalized(reach);
  // Along the shoulder-hand line the elbow sits at `a`; off it by `h`.
  const float a = (upper * upper - fore * fore + d * d) / (2 * d);
  const float h = std::sqrt(std::max(upper * upper - a * a, 0.0f));
  Vec3 bend = Vec3{0, -1, 0} - dir * dot(Vec3{0, -1, 0}, dir);
  bend = length(bend) > 1e-3f ? normalized(bend) : Vec3{1, 0, 0};
  const Vec3 elbow = shoulder + dir * a + bend * h;
  r.draw(Mesh::Cylinder, render::along(shoulder, elbow, radius), sleeve, Material::Matte);
  r.draw(Mesh::Sphere, render::translation(elbow) * render::scaling({radius, radius, radius}), skin,
         Material::Matte);
  r.draw(Mesh::Cylinder, render::along(elbow, hand, radius * 0.85f), skin, Material::Matte);
  r.draw(Mesh::Sphere,
         render::translation(hand) * render::scaling({radius * 1.15f, radius * 1.15f, radius * 1.15f}),
         skin, Material::Matte);
}

// The opponent: a plain figure standing behind its bat. It gives the far end
// of the table a person to play against, and its size is a depth cue.
void draw_opponent(MetalRenderer& r, const Paddle& bat, float body_x) {
  const Color shirt = rgb(44, 96, 168), shorts = rgb(28, 30, 38), skin = rgb(190, 168, 150);
  const float z = bat.pos.z - 0.42f;
  for (const float side : {-1.0f, 1.0f}) {
    r.draw(Mesh::Cylinder,
           render::along({body_x + side * 0.10f, 0, z}, {body_x + side * 0.09f, 0.90f, z}, 0.062f),
           side < 0 ? shorts : shorts, Material::Matte);
  }
  r.draw(Mesh::Sphere, render::translation({body_x, 1.16f, z}) * render::scaling({0.20f, 0.30f, 0.12f}),
         shirt, Material::Matte);
  r.draw(Mesh::Sphere, render::translation({body_x, 1.60f, z}) * render::scaling({0.10f, 0.115f, 0.10f}),
         skin, Material::Matte);

  const Vec3 grip = draw_bat(r, bat, rgb(52, 110, 190), rgb(28, 28, 32), 1.0f);
  // Its right arm holds the bat (on our left); the other hangs ready.
  draw_arm(r, {body_x - 0.21f, 1.38f, z + 0.02f}, grip, 0.30f, 0.30f, 0.045f, shirt, skin);
  draw_arm(r, {body_x + 0.21f, 1.38f, z + 0.02f}, {body_x + 0.30f, 0.98f, z + 0.22f}, 0.28f, 0.27f,
           0.045f, shirt, skin);
}

}  // namespace

// ---------------------------------------------------------------------- app

struct GameApp::Impl {
  MetalRenderer& r;
  AppOptions options;

  // Order matters. The tracker is declared after the simulation, so it is
  // destroyed first: its destructor waits for the last camera callback, and
  // that callback pushes into the simulation's queue.
  SimThread sim{&HandTracker::now};
  HandTracker tracker;
  SessionLog session;
  Sounds sounds;

  bool quit = false, camera_wanted = false, show_details = true;
  bool assist = true, magnet = true;
  real speed = kDefaultSimSpeed;
  int difficulty = kDefaultDifficulty;
  Vec3 pointer = on_surface(0, 1.0f);
  float mouse_x = 640, mouse_y = 360;

  // What was last seen in the snapshot, to notice events.
  std::uint32_t paddle_hits = 0, table_hits = 0, net_hits = 0, floor_hits = 0, player_hits = 0,
                swings = 0;
  std::uint64_t last_tick = 0, hands_seen = 0, preview_seq = 0;
  float sim_rate = 0, fps = 0;
  double camera_fps = 0, last_capture = 0, capture_to_display_ms = 0;
  double clock = 0, last_flush = 0;
  int frame_no = 0, bench_seen = 0, latency_samples = 0;
  double deliver_ms = 0, detect_ms = 0;  // camera delay, summed over the frames seen

  // Animation state.
  float hit_flash = 0, hit_power = 0, shake = 0, swing_anim = 1, target_mark = 0;
  Vec3 swing_anim_dir{0, 0, 1};
  float opponent_x = 0, eye_x = 0;
  PreviewImage preview;
  std::array<float, 240> frame_ms{};
  std::size_t frame_head = 0;

  Impl(MetalRenderer& renderer, const AppOptions& opt) : r(renderer), options(opt) {}

  void set_camera(bool on) {
    camera_wanted = on;
    if (on) tracker.start([this](const HandObservation& obs) { sim.push_hand(obs); });
    else tracker.stop();
    sim.send({.kind = SimCommand::Kind::UseHand, .flag = on});
  }

  void centered(std::string_view text, float y, float size, Color color) {
    r.text((r.width() - r.text_width(size, text)) * 0.5f, y, size, color, text);
  }

  void play_sounds(const Snapshot& snap);
  void draw_scene(const Snapshot& snap, float dt);
  void draw_hud(const Snapshot& snap, float dt);
  void draw_camera_overlay(const Snapshot& snap);
};

GameApp::GameApp(MetalRenderer& renderer, const AppOptions& options)
    : impl_(std::make_unique<Impl>(renderer, options)) {
  if (options.record && options.screenshot == nullptr) {
    impl_->session.open(SessionLog::default_root());
  }
  if (options.camera) impl_->set_camera(true);
}

GameApp::~GameApp() {
  impl_->tracker.stop();
  if (impl_->session.is_open()) {
    impl_->session.close();
    std::printf("session recorded in %s\n", impl_->session.directory().c_str());
  }
}

bool GameApp::wants_quit() const { return impl_->quit; }

void GameApp::mouse_moved(float x, float y) {
  impl_->mouse_x = x;
  impl_->mouse_y = y;
}

void GameApp::key_pressed(char key) {
  Impl& s = *impl_;
  using Kind = SimCommand::Kind;
  switch (key) {
    case 'r': s.sim.send({.kind = Kind::Restart}); break;
    case 'a':
      s.assist = !s.assist;
      s.sim.send({.kind = Kind::SetAssist, .flag = s.assist});
      break;
    case '[':
    case ']':
      s.speed = std::clamp(s.speed + (key == '[' ? -0.05f : 0.05f), 0.40f, 1.00f);
      s.sim.send({.kind = Kind::SetSpeed, .value = s.speed});
      break;
    case 'c': s.set_camera(!s.camera_wanted); break;
    case 'k': s.sim.send({.kind = Kind::Recentre}); break;
    case 'm':
      s.magnet = !s.magnet;
      s.sim.send({.kind = Kind::SetMagnet, .flag = s.magnet});
      break;
    case 'd':
      s.difficulty = (s.difficulty + 1) % kDifficultyLevels;
      s.sim.send({.kind = Kind::SetDifficulty, .value = static_cast<real>(s.difficulty)});
      break;
    case 'h': s.show_details = !s.show_details; break;
    case 27: s.quit = true; break;  // escape
    default: break;
  }
}

void GameApp::Impl::play_sounds(const Snapshot& snap) {
  const float side = std::clamp(snap.ball.pos.x / 0.76f, -1.0f, 1.0f);
  const float nearness = std::clamp((snap.ball.pos.z + 1.6f) / 3.1f, 0.0f, 1.0f);
  const float leaving = length(snap.ball.vel);
  if (snap.paddle_hits != paddle_hits) sounds.bat(std::max(snap.impact_speed, leaving), side, nearness);
  if (snap.table_hits != table_hits) sounds.table(snap.impact_speed, side, nearness);
  if (snap.net_hits != net_hits) sounds.net(snap.impact_speed);
  if (snap.floor_hits != floor_hits) sounds.floor(snap.impact_speed, side);
  paddle_hits = snap.paddle_hits;
  table_hits = snap.table_hits;
  net_hits = snap.net_hits;
  floor_hits = snap.floor_hits;
}

void GameApp::Impl::draw_scene(const Snapshot& snap, float dt) {
  const Paddle& me = snap.paddles[index(Side::Player)];
  const Paddle& them = snap.paddles[index(Side::Opponent)];

  // ---- the eye ----
  // It leans a little with the bat, as a player's head does, and a hard hit
  // is felt as a small jolt.
  // The player's body goes to the ball by itself; the eye goes with it (half
  // way, so the table does not swim), which is what makes that read as
  // stepping across and not as the bat sliding off on its own.
  const float stand = camera_wanted ? 0.5f * snap.body_x : 0.10f * me.pos.x;
  eye_x += (stand - eye_x) * (1.0f - std::exp(-dt / 0.12f));
  if (snap.player_hits != player_hits) {
    shake = std::clamp(snap.last_hit_speed / 11.0f, 0.2f, 1.0f);
    target_mark = 0.8f;
  }
  shake *= std::exp(-dt / 0.07f);
  const float t = static_cast<float>(clock);
  const Vec3 jolt = Vec3{std::sin(t * 91.0f), std::cos(t * 77.0f), 0} * (0.012f * shake);
  render::CameraSetup camera;
  camera.eye = Vec3{eye_x, 1.55f, 3.25f} + jolt;
  camera.target = Vec3{eye_x * 0.55f, 0.82f, 0} + jolt * 0.5f;
  r.begin_frame(camera);

  draw_hall(r);
  draw_table(r, Table{}, Net{});

  opponent_x += (0.8f * them.pos.x - opponent_x) * (1.0f - std::exp(-dt / 0.18f));
  draw_opponent(r, them, opponent_x);

  // ---- the ball ----
  // A camera's shutter is open for a moment, so a fast ball is a streak, not
  // a dot. Drawn as fading copies back along its path.
  const Vec3 ball = snap.ball.pos;
  const float rad = snap.ball_radius;
  r.draw(Mesh::Sphere, render::translation(ball) * render::scaling({rad, rad, rad}), kAmber,
         Material::Ball);
  const Vec3 streak = snap.ball.vel * (snap.speed * 0.030f);  // 30 ms of travel, in real time
  if (length(streak) > rad && snap.phase == Phase::Rally) {
    for (int k = 1; k <= 5; ++k) {
      const float f = static_cast<float>(k) / 6.0f;
      const float size = rad * (1.0f - 0.35f * f);
      r.draw(Mesh::Sphere, render::translation(ball - streak * f) * render::scaling({size, size, size}),
             render::fade(kAmber, 0.30f * (1.0f - f)), Material::Ball, false);
    }
  }

  // ---- where the player's shot was sent ----
  // A mark on the far half, for a moment after each hit: swing right and it
  // appears on the right. It is what lets the player see that the shot went
  // where the swing said, and learn the swing that sends it elsewhere.
  if (target_mark > 0) {
    target_mark -= dt;
    const float size = 0.07f;
    r.draw(Mesh::Ring,
           render::basis({size, 0, 0}, {0, 0, -size}, {0, size * 0.3f, 0},
                         {snap.last_target.x, Table{}.height + 0.004f, snap.last_target.z}),
           render::fade(kAmber, std::clamp(target_mark / 0.4f, 0.0f, 0.9f)), Material::Glow, false);
  }

  // ---- where to be, and when to swing ----
  if (snap.has_ring) {
    const Color ring = snap.ring_good ? kGreen : kRed;
    r.draw(Mesh::Ring, render::translation(snap.ring) * render::scaling({me.radius, me.radius, me.radius}),
           render::fade(ring, 0.85f), Material::Glow, false);
    // Timing cue: an outer circle closes in on the ring as the ball
    // approaches. Swing as the two meet.
    if (snap.ring_good && snap.ball_due_in >= 0) {
      const float closing = std::clamp(snap.ball_due_in / 0.7f, 0.0f, 1.0f);
      const float size = me.radius * (1.0f + 2.4f * closing);
      r.draw(Mesh::Ring, render::translation(snap.ring) * render::scaling({size, size, size * 0.5f}),
             render::fade(kWhite, 0.55f), Material::Glow, false);
    }
  }

  // ---- the player's bat ----
  // With the camera, the bat is drawn where the hand has it, swing and all:
  // the simulation publishes the hand's own travel. With the mouse there is
  // no swing to show, so one is animated.
  if (snap.swings != swings) {
    swings = snap.swings;
    swing_anim = 0;
  }
  if (snap.swinging) swing_anim_dir = snap.swing_dir;
  swing_anim = std::min(1.0f, swing_anim + dt / 0.32f);
  Paddle shown = me;
  if (!camera_wanted && swing_anim < 1.0f) {
    const float out = std::sin(kPi * std::sqrt(swing_anim));  // out fast, back slower
    const Vec3 d = swing_anim_dir;
    shown.pos += Vec3{0.16f * d.x, 0.16f * d.y, -0.14f - 0.10f * std::abs(d.z)} * out;
  }
  // Slightly see-through, so the ball can be watched right onto the bat.
  const float alpha = swing_anim < 1.0f ? 0.95f : 0.82f;
  draw_bat(r, shown, rgb(28, 28, 32), rgb(190, 44, 40), alpha, camera_wanted,
           snap.paddle_roll);
}

void GameApp::Impl::draw_camera_overlay(const Snapshot& snap) {
  const CameraStatus status = tracker.status();
  const bool running = status == CameraStatus::Running;
  tracker.copy_preview(preview, preview_seq);

  const float w = 260, h = preview.width > 0 ? w * static_cast<float>(preview.height) /
                                                   static_cast<float>(preview.width)
                                             : 195;
  const float x0 = r.width() - w - 16, y0 = r.height() - h - 44;
  r.rect(x0 - 2, y0 - 40, w + 4, h + 42, rgb(0, 0, 0, 0.6f));
  std::array<char, 160> buf;
  r.text(x0 + 4, y0 - 37, 14, kLight,
         running ? fmt(buf, "Camera %.0f fps   detect %.0f ms   dropped %llu", camera_fps,
                       1000.0 * (snap.hand.detect_time - snap.hand.capture_time),
                       static_cast<unsigned long long>(snap.hands_dropped))
                 : fmt(buf, "Camera: %s", describe(status)));
  if (running) {
    r.text(x0 + 4, y0 - 19, 14, snap.hand_in_control ? kGreen : kGray,
           snap.hand_in_control ? "Hand in control. K sets this spot as centre."
                                : "Show an open palm to take control");
  }
  if (!preview.rgba.empty()) r.image(preview.rgba.data(), preview.width, preview.height, x0, y0, w, h);
  if (!snap.hand.found) return;

  const auto at = [&](const JointPoint& j, float& x, float& y) {
    x = x0 + j.x * w;
    y = y0 + (1.0f - j.y) * h;
  };
  // Bones: wrist to the base of each finger, then along the finger.
  for (std::size_t finger = 0; finger < 5; ++finger) {
    const JointPoint* prev = &snap.hand.joints[0];
    for (std::size_t k = 0; k < 4; ++k) {
      const JointPoint& j = snap.hand.joints[1 + finger * 4 + k];
      if (prev->confidence > 0.3f && j.confidence > 0.3f) {
        float ax, ay, bx, by;
        at(*prev, ax, ay);
        at(j, bx, by);
        r.line(ax, ay, bx, by, 2.0f, rgb(102, 191, 255, 0.9f));
      }
      prev = &j;
    }
  }
  for (const JointPoint& j : snap.hand.joints) {
    if (j.confidence <= 0.3f) continue;
    float x, y;
    at(j, x, y);
    r.circle(x, y, 3.0f, kAmber);
  }
  if (snap.palm.valid) {
    float x, y;
    at({snap.palm.x, snap.palm.y, 1}, x, y);
    r.circle_outline(x, y, 14.0f, 1.5f, kGreen);
  }
}

void GameApp::Impl::draw_hud(const Snapshot& snap, float dt) {
  std::array<char, 160> buf;
  const int you = snap.score[index(Side::Player)], them = snap.score[index(Side::Opponent)];
  const float cx = r.width() * 0.5f;

  // ---- scoreboard ----
  r.rect(cx - 150, 10, 300, 76, rgb(0, 0, 0, 0.45f));
  centered(fmt(buf, "%d   :   %d", you, them), 12, 46, kWhite);
  r.text(cx - 132, 62, 13, kGray, "YOU");
  r.text(cx + 132 - r.text_width(13, "OPPONENT"), 62, 13, kGray, "OPPONENT");
  if (snap.phase != Phase::GameOver) {  // a dot beside whoever is serving
    r.circle(cx + (snap.server == Side::Player ? -92.0f : 92.0f), 38, 5, kAmber);
  }

  if (snap.phase == Phase::GameOver) {
    centered(you > them ? "You win the game" : "Opponent wins the game", 110, 34, kWhite);
    centered("Press R to play again", 152, 18, kGray);
  } else if (snap.phase == Phase::PointOver && snap.last_reason != PointReason::None) {
    const bool let = snap.last_reason == PointReason::Let;
    const bool mine = snap.last_winner == Side::Player;
    centered(let ? "" : mine ? "Your point" : "Opponent's point", 104, 26, mine ? kGreen : kRed);
    centered(describe(snap.last_reason), 136, 18, kLight);
  } else if (snap.server == Side::Player &&
             (snap.phase == Phase::PreServe || snap.stage == Stage::ServeToss)) {
    centered("Your serve: swing as the ball comes back down", 108, 18, kLight);
  }

  // ---- a struck ball says so, with what kind of stroke it was ----
  if (snap.player_hits != player_hits) {
    player_hits = snap.player_hits;
    hit_flash = 0.6f;
    hit_power = snap.swing_power;
  }
  if (hit_flash > 0) {
    hit_flash -= dt;
    const char* word = snap.last_stroke == StrokeKind::Topspin
                           ? (hit_power > 0.66f ? "TOPSPIN LOOP" : "TOPSPIN")
                       : snap.last_stroke == StrokeKind::Chop ? "CHOP"
                       : hit_power > 0.66f ? "SMASH" : hit_power > 0.25f ? "DRIVE" : "TAP";
    const float fade_out = std::clamp(hit_flash / 0.3f, 0.0f, 1.0f);
    centered(word, r.height() * 0.5f + 150, 30, render::fade(kAmber, fade_out));
    // And how it was timed, when it was not timed well.
    if (snap.last_timing != StrokeTiming::Good) {
      centered(snap.last_timing == StrokeTiming::Late ? "late" : "early", r.height() * 0.5f + 186, 18,
               render::fade(kRed, fade_out));
    }
  }

  if (!show_details) return;

  // ---- settings, bottom left ----
  const float bottom = r.height();
  r.text(16, bottom - 50, 15, kLight,
         "SWING as the white circle closes on the ring. The ball goes the way you swing: left,"
         " right, up = topspin, down = chop. Harder = faster.");
  r.text(16, bottom - 28, 15, kGray,
         "C camera   K recentre   D opponent   M magnet   A aim assist   [ ] speed   H hide   R restart");
  r.text(16, 14, 15, kGray, fmt(buf, "Opponent: %s", difficulty_name(snap.difficulty)));
  r.text(16, 34, 15, kGray, fmt(buf, "Speed: %d%%", static_cast<int>(std::lround(snap.speed * 100.0f))));
  r.text(16, 54, 15, snap.magnet ? kGreen : kGray, snap.magnet ? "Magnet: on" : "Magnet: off");
  r.text(16, 74, 15, snap.assist ? kGreen : kGray, snap.assist ? "Aim assist: on" : "Aim assist: off");
  if (session.is_open()) r.text(16, 94, 13, kGray, "Recording play data");

  // ---- the pipeline's vital signs, top right ----
  const float x = r.width() - 330;
  r.rect(x - 4, 12, 248, 68, rgb(0, 0, 0, 0.5f));
  for (std::size_t i = 0; i < frame_ms.size(); ++i) {
    const float v = frame_ms[(frame_head + i) % frame_ms.size()];
    const float bar = std::min(60.0f, v * 2.0f);  // 2 points per millisecond
    r.rect(x + static_cast<float>(i), 76 - bar, 1, bar, v > 17.0f ? rgb(255, 161, 0) : rgb(102, 191, 255));
  }
  const render::FrameStats stats = r.stats();
  r.text(x, 86, 13, kGray, fmt(buf, "%.0f fps   Metal: cpu %.2f ms, gpu %.2f ms, %d draws",
                               static_cast<double>(fps), static_cast<double>(stats.cpu_ms),
                               static_cast<double>(stats.gpu_ms), stats.draws));
  r.text(x, 104, 13, kGray, fmt(buf, "sim %.0f ticks/s   load %.1f%%", static_cast<double>(sim_rate),
                                static_cast<double>(snap.busy_fraction) * 100.0));
  r.text(x, 122, 13, kGray, fmt(buf, "wake late %.0f us avg, %.0f max",
                                static_cast<double>(snap.wake_late_mean_us),
                                static_cast<double>(snap.wake_late_max_us)));
  if (camera_wanted && capture_to_display_ms > 0) {
    r.text(x, 140, 13, kAmber, fmt(buf, "camera to screen %.0f ms", capture_to_display_ms));
  }
  if (camera_wanted) draw_camera_overlay(snap);
}

void GameApp::frame(float dt) {
  Impl& s = *impl_;
  PP_INTERVAL("frame");
  dt = std::clamp(dt, 0.0f, 0.25f);
  s.clock += static_cast<double>(dt);
  s.frame_ms[s.frame_head] = dt * 1000.0f;
  s.frame_head = (s.frame_head + 1) % s.frame_ms.size();
  if (dt > 0) s.fps += (1.0f / dt - s.fps) * 0.05f;

  // ---- the newest state of the world ----
  // A copy: one consistent picture for the whole frame.
  const Snapshot snap = s.sim.latest();
  s.play_sounds(snap);

  // Drain the simulation's event queue into the session file. Disk writes
  // belong here, on the thread that can afford a hiccup.
  TelemetryEvent event;
  while (s.sim.poll_telemetry(event)) s.session.write(event);
  if (s.clock - s.last_flush > 2.0) {
    s.session.flush();
    s.last_flush = s.clock;
  }

  if (dt > 0) {
    const float rate = static_cast<float>(snap.tick - s.last_tick) / dt;
    s.sim_rate += (rate - s.sim_rate) * 0.05f;
  }
  s.last_tick = snap.tick;

  if (s.camera_wanted) {
    if (snap.hands_received != s.hands_seen && s.last_capture > 0) {
      // Frame rate actually delivered, from the camera's own timestamps.
      const double gap = snap.hand.capture_time - s.last_capture;
      if (gap > 0) {
        s.camera_fps +=
            (static_cast<double>(snap.hands_received - s.hands_seen) / gap - s.camera_fps) * 0.1;
      }
    }
    // (Skipping the first two seconds: the first detections load the model.)
    if (snap.hands_received != s.hands_seen && s.clock > 2.0) {
      s.deliver_ms += 1000.0 * (snap.hand.receive_time - snap.hand.capture_time);
      s.detect_ms += 1000.0 * (snap.hand.detect_time - snap.hand.receive_time);
      ++s.latency_samples;
    }
    s.hands_seen = snap.hands_received;
    s.last_capture = snap.hand.capture_time;
  }

  s.draw_scene(snap, dt);  // begins the frame, and sets the camera the mouse ray needs
  s.draw_hud(snap, dt);

  // ---- mouse: where its ray meets the surface the mouse bat lives on ----
  if (s.options.demo) {  // unattended: wave the bat so that rallies happen
    s.mouse_y = s.r.height() * 0.5f + 140.0f * std::sin(static_cast<float>(s.clock) * 13.0f);
    s.mouse_x = s.r.width() * 0.5f + 160.0f * std::sin(static_cast<float>(s.clock) * 1.7f);
  }
  {
    Vec3 origin, dir;
    s.r.ray_through(s.mouse_x, s.mouse_y, origin, dir);
    const Vec3 n = surface_toward_ball();
    const real denom = dot(dir, n);
    if (denom >= 1e-4f) {
      const real t = dot(kSurfacePoint - origin, n) / denom;
      s.pointer = on_surface(origin.x + t * dir.x, origin.y + t * dir.y);
    }
    s.sim.send({.kind = SimCommand::Kind::PointerTarget, .target = s.pointer});
  }

  const bool shoot = s.options.screenshot != nullptr && ++s.frame_no == s.options.screenshot_frame;
  s.r.end_frame(shoot ? s.options.screenshot : nullptr);
  if (shoot) s.quit = true;
  if (s.options.bench_frames > 0 && ++s.bench_seen == s.options.bench_frames) {
    const render::FrameStats stats = s.r.stats();
    std::printf("after %d frames: %.0f fps, frame cpu %.3f ms, gpu %.3f ms, %d draws\n",
                s.bench_seen, static_cast<double>(s.fps), static_cast<double>(stats.cpu_ms),
                static_cast<double>(stats.gpu_ms), stats.draws);
    if (s.latency_samples > 0) {
      std::printf("camera: exposure to delivery %.1f ms, hand detection %.1f ms (%d frames, hand %s)\n",
                  s.deliver_ms / s.latency_samples, s.detect_ms / s.latency_samples,
                  s.latency_samples, snap.hand.found ? "in view" : "not in view");
    }
    s.quit = true;
  }

  // End-to-end latency: from the camera exposing the frame to the picture
  // that used it being handed to the display.
  if (s.camera_wanted && snap.hand_in_control) {
    const double ms = 1000.0 * (HandTracker::now() - snap.hand.capture_time);
    s.capture_to_display_ms += (ms - s.capture_to_display_ms) * 0.05;
  }
}

}  // namespace pp
