// Palm Rally: one game of table tennis, played with the mouse or with your
// palm in front of the camera.
//
// Three threads:
//   capture thread   camera frames -> hand pose         (src/platform)
//   sim thread       physics and rules at 1 kHz         (src/game/sim_thread)
//   main thread      this file: input, drawing, sound
// The main thread never touches the simulation. It sends small commands in
// and draws whatever snapshot the simulation published last.
#include <raylib.h>

#include "core/signpost.hpp"
#include "game/paddle_surface.hpp"
#include "game/session_log.hpp"
#include "game/sim_thread.hpp"
#include "platform/hand_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <string_view>

namespace {

using pp::real;
using pp::Vec3;

constexpr float kMaxFrame = 0.25f;

Vector3 rl(Vec3 v) { return {v.x, v.y, v.z}; }

// Where the mouse ray meets the paddle surface.
Vec3 mouse_on_surface(const Camera3D& camera, Vec3 fallback) {
  const Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
  const Vec3 origin{ray.position.x, ray.position.y, ray.position.z};
  const Vec3 dir{ray.direction.x, ray.direction.y, ray.direction.z};
  const Vec3 n = pp::surface_toward_ball();
  const real denom = dot(dir, n);
  if (denom < 1e-4f) return fallback;  // looking away from the surface
  const real t = dot(pp::kSurfacePoint - origin, n) / denom;
  return pp::on_surface(origin.x + t * dir.x, origin.y + t * dir.y);
}

// ------------------------------------------------------------------ camera

// The camera image with the detected hand drawn over it.
void draw_camera_overlay(const Texture2D& image, const pp::HandObservation& hand,
                         const pp::Palm& palm, const char* caption, const char* hint,
                         Color hint_color) {
  const int w = 320, h = image.id != 0 ? image.height * 320 / std::max(1, image.width) : 240;
  const int x0 = GetScreenWidth() - w - 16, y0 = GetScreenHeight() - h - 44;
  DrawRectangle(x0 - 2, y0 - 40, w + 4, h + 42, Fade(BLACK, 0.6f));
  DrawText(caption, x0 + 4, y0 - 36, 14, LIGHTGRAY);
  DrawText(hint, x0 + 4, y0 - 18, 14, hint_color);
  if (image.id != 0) {
    DrawTexturePro(image, {0, 0, static_cast<float>(image.width), static_cast<float>(image.height)},
                   {static_cast<float>(x0), static_cast<float>(y0), static_cast<float>(w),
                    static_cast<float>(h)},
                   {0, 0}, 0, WHITE);
  }
  if (!hand.found) return;

  const auto at = [&](const pp::JointPoint& j) {
    return Vector2{static_cast<float>(x0) + j.x * static_cast<float>(w),
                   static_cast<float>(y0) + (1.0f - j.y) * static_cast<float>(h)};
  };
  // Bones: wrist to the base of each finger, then along the finger.
  for (std::size_t finger = 0; finger < 5; ++finger) {
    const pp::JointPoint* prev = &hand.joints[0];
    for (std::size_t k = 0; k < 4; ++k) {
      const pp::JointPoint& j = hand.joints[1 + finger * 4 + k];
      if (prev->confidence > 0.3f && j.confidence > 0.3f) {
        DrawLineEx(at(*prev), at(j), 2.0f, Fade(SKYBLUE, 0.9f));
      }
      prev = &j;
    }
  }
  for (const pp::JointPoint& j : hand.joints) {
    if (j.confidence > 0.3f) DrawCircleV(at(j), 3.0f, {250, 176, 60, 255});
  }
  if (palm.valid) {
    DrawCircleLinesV(at({palm.x, palm.y, 1}), 14.0f, {96, 196, 150, 255});
  }
}

// ------------------------------------------------------------------- sound

// A short decaying tone, synthesised so the project needs no asset files.
Sound make_click(float frequency, float seconds, float volume) {
  constexpr int kRate = 44100;
  const int frames = static_cast<int>(static_cast<float>(kRate) * seconds);
  auto* samples = static_cast<short*>(MemAlloc(static_cast<unsigned>(frames) * sizeof(short)));
  for (int i = 0; i < frames; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(kRate);
    const float envelope = std::exp(-t * 6.0f / seconds);
    const float wave = std::sin(2.0f * std::numbers::pi_v<float> * frequency * t) +
                       0.4f * std::sin(2.0f * std::numbers::pi_v<float> * frequency * 2.7f * t);
    samples[i] = static_cast<short>(wave * envelope * volume * 20000.0f);
  }
  const Wave wave{.frameCount = static_cast<unsigned>(frames),
                  .sampleRate = kRate,
                  .sampleSize = 16,
                  .channels = 1,
                  .data = samples};
  const Sound sound = LoadSoundFromWave(wave);
  UnloadWave(wave);
  return sound;
}

// ----------------------------------------------------------------- drawing

void draw_table(const pp::Table& t, const pp::Net& net) {
  const Color surface{24, 78, 104, 255};
  const Color line{235, 240, 242, 255};
  DrawCube({0, t.height - 0.015f, 0}, t.width, 0.03f, t.length, surface);
  DrawCube({0, t.height + 0.0005f, 0}, 0.004f, 0.001f, t.length, line);
  DrawCubeWires({0, t.height - 0.015f, 0}, t.width, 0.03f, t.length, line);
  DrawCube({0, t.height + net.height / 2, 0}, t.width + 2 * net.overhang, net.height, 0.004f,
           Fade(line, 0.55f));
}

// A bat: wooden blade edge, rubber on both faces, and a handle. The handle
// hangs down and leans toward the body, so it swaps sides as the paddle
// crosses the middle, the way a forehand turns into a backhand.
// With a tracked hand, `roll` turns the bat the way the hand is turned.
void draw_bat(const pp::Paddle& p, Color front, Color back, float alpha, bool use_roll = false,
              real roll = 0) {
  const Color wood{196, 154, 98, 255};
  const Vec3 n = p.normal;

  // Fingers point up along the blade, so the handle points the opposite way.
  Vec3 down = use_roll ? Vec3{-std::sin(roll), -std::cos(roll), 0}
                       : Vec3{std::clamp(-p.pos.x * 1.2f, -0.6f, 0.6f), -1, 0};
  down = pp::normalized(down - dot(down, n) * n);  // lie in the blade's plane

  DrawCylinderEx(rl(p.pos + down * (p.radius * 0.85f)), rl(p.pos + down * (p.radius + 0.105f)),
                 0.015f, 0.013f, 12, Fade(wood, alpha));
  DrawCylinderEx(rl(p.pos - n * 0.004f), rl(p.pos + n * 0.004f), p.radius, p.radius, 32,
                 Fade(wood, alpha));
  DrawCylinderEx(rl(p.pos + n * 0.004f), rl(p.pos + n * 0.007f), p.radius * 0.97f,
                 p.radius * 0.97f, 32, Fade(front, alpha));
  DrawCylinderEx(rl(p.pos - n * 0.007f), rl(p.pos - n * 0.004f), p.radius * 0.97f,
                 p.radius * 0.97f, 32, Fade(back, alpha));
}

void draw_ball(Vec3 b, real radius) {
  const pp::Table t;
  const bool over_table = std::abs(b.x) <= t.width / 2 && std::abs(b.z) <= t.length / 2 &&
                          b.y >= t.height;
  const float ground = over_table ? t.height + 0.001f : 0.001f;
  if (b.y >= ground) {  // the shadow is the main depth cue, keep it crisp
    DrawCylinder({b.x, ground, b.z}, radius, radius, 0.0005f, 16, Fade(BLACK, 0.45f));
  }
  DrawSphere(rl(b), radius, {250, 176, 60, 255});
}

// Recent ball positions, drawn as a fading tail so a fast ball reads as motion.
struct Trail {
  std::array<Vec3, 14> points{};
  std::size_t head = 0, count = 0;

  void push(Vec3 p) {
    points[head] = p;
    head = (head + 1) % points.size();
    count = std::min(count + 1, points.size());
  }
  void draw(real radius) const {
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t age = i + 1;  // 1 = newest
      const Vec3 p = points[(head + points.size() - age) % points.size()];
      const float k = 1.0f - static_cast<float>(age) / static_cast<float>(points.size() + 1);
      DrawSphereEx(rl(p), radius * (0.35f + 0.5f * k), 6, 8, Fade({250, 176, 60, 255}, 0.30f * k));
    }
  }
};

// The target ring: where to hold the paddle. Drawn in the surface's own plane.
void draw_ring(Vec3 at, float radius, Color color) {
  const float tilt = std::atan(pp::kSurfaceSlope) * RAD2DEG;
  DrawCircle3D(rl(at), radius, {1, 0, 0}, tilt, color);
  DrawCircle3D(rl(at), radius * 0.985f, {1, 0, 0}, tilt, color);
  DrawCircle3D(rl(at), 0.02f, {1, 0, 0}, tilt, color);
}

void draw_frame_graph(const std::array<float, 240>& ms, std::size_t head) {
  const int x0 = GetScreenWidth() - 256, y0 = 76, h = 60;
  DrawRectangle(x0 - 4, y0 - h - 4, static_cast<int>(ms.size()) + 8, h + 8, Fade(BLACK, 0.5f));
  for (std::size_t i = 0; i < ms.size(); ++i) {
    const float v = ms[(head + i) % ms.size()];
    const int bar = std::min(h, static_cast<int>(v * 2.0f));  // 2 px per ms
    DrawLine(x0 + static_cast<int>(i), y0, x0 + static_cast<int>(i), y0 - bar,
             v > 17.0f ? ORANGE : SKYBLUE);
  }
}

void draw_centered(const char* text, int y, int size, Color color) {
  DrawText(text, (GetScreenWidth() - MeasureText(text, size)) / 2, y, size, color);
}

void draw_hud(const pp::Snapshot& m, bool recording) {
  using pp::Phase;
  using pp::Side;
  const int you = m.score[pp::index(Side::Player)], them = m.score[pp::index(Side::Opponent)];

  draw_centered(TextFormat("%d   :   %d", you, them), 14, 44, RAYWHITE);
  draw_centered("YOU                    OPPONENT", 62, 14, GRAY);
  const int cx = GetScreenWidth() / 2;
  if (m.phase != Phase::GameOver) {  // a dot under whoever is serving
    DrawCircle(cx + (m.server == Side::Player ? -92 : 92), 88, 5, {250, 176, 60, 255});
  }

  if (m.phase == Phase::GameOver) {
    draw_centered(you > them ? "You win the game" : "Opponent wins the game", 130, 34, RAYWHITE);
    draw_centered("Press R to play again", 172, 18, GRAY);
  } else if (m.phase == Phase::PointOver && m.last_reason != pp::PointReason::None) {
    const bool let = m.last_reason == pp::PointReason::Let;
    const char* who = let ? "" : m.last_winner == Side::Player ? "Your point" : "Opponent's point";
    draw_centered(who, 126, 26, m.last_winner == Side::Player ? Color{96, 196, 150, 255}
                                                              : Color{214, 92, 80, 255});
    draw_centered(pp::describe(m.last_reason), 158, 18, LIGHTGRAY);
  } else if (m.server == Side::Player &&
             (m.phase == Phase::PreServe || m.stage == pp::Stage::ServeToss)) {
    draw_centered("Your serve: swing as the ball comes back down", 130, 18, LIGHTGRAY);
  }

  DrawText("SWING to hit: a quick flick as the white circle closes on the ring. Harder = faster,"
           " sideways = aim.",
           16, GetScreenHeight() - 50, 16, LIGHTGRAY);
  DrawText("C: camera    K: recentre hand    M: magnet    A: aim assist    [ ]: speed    R: restart",
           16, GetScreenHeight() - 28, 16, GRAY);
  DrawText(TextFormat("Magnet: %s", m.magnet ? "on" : "off"), 16, 58, 16,
           m.magnet ? Color{96, 196, 150, 255} : GRAY);
  if (recording) DrawText("Recording play data", 16, 80, 14, GRAY);
  DrawText(TextFormat("Aim assist: %s", m.assist ? "on" : "off"), 16, 14, 16,
           m.assist ? Color{96, 196, 150, 255} : GRAY);
  DrawText(TextFormat("Speed: %d%%", static_cast<int>(std::lround(m.speed * 100.0f))), 16, 36, 16,
           GRAY);
}

// The pipeline's vital signs, top right under the frame graph.
void draw_timing(const pp::Snapshot& m, float sim_rate, double capture_to_display_ms) {
  const int x = GetScreenWidth() - 256;
  DrawText(TextFormat("%d fps", GetFPS()), x, 86, 14, GRAY);
  DrawText(TextFormat("sim %.0f ticks/s   load %.1f%%", static_cast<double>(sim_rate),
                      static_cast<double>(m.busy_fraction) * 100.0),
           x, 104, 14, GRAY);
  DrawText(TextFormat("wake late %.0f us avg, %.0f max",
                      static_cast<double>(m.wake_late_mean_us),
                      static_cast<double>(m.wake_late_max_us)),
           x, 122, 14, GRAY);
  if (capture_to_display_ms > 0) {
    DrawText(TextFormat("camera to screen %.0f ms", capture_to_display_ms), x, 140, 14,
             {250, 176, 60, 255});
  }
}

}  // namespace

int main(int argc, char** argv) {
  // --screenshot <file.png>: render for a while, save one frame, exit.
  // Lets a script check the scene without a person watching.
  // --camera: start with the camera on.
  // --no-record: do not write a session log.
  const char* screenshot_path = nullptr;
  bool camera_wanted = false, record = true;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--screenshot" && i + 1 < argc) screenshot_path = argv[i + 1];
    if (arg == "--camera") camera_wanted = true;
    if (arg == "--no-record") record = false;
  }
  if (screenshot_path != nullptr) record = false;  // scripted runs are not play data
  int frame_no = 0;

  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_WINDOW_HIGHDPI);
  InitWindow(1280, 720, "Palm Rally");
  InitAudioDevice();
  const Sound paddle_sound = make_click(820.0f, 0.07f, 0.9f);
  const Sound table_sound = make_click(1450.0f, 0.045f, 0.6f);
  const Sound net_sound = make_click(240.0f, 0.09f, 0.5f);

  Camera3D camera{};
  camera.position = {0.0f, 1.55f, 3.25f};
  camera.target = {0.0f, 0.82f, 0.0f};
  camera.up = {0.0f, 1.0f, 0.0f};
  camera.fovy = 45.0f;
  camera.projection = CAMERA_PERSPECTIVE;

  // Order matters. The tracker is declared after the simulation, so it is
  // destroyed first: its destructor waits for the last camera callback, and
  // that callback pushes into the simulation's queue.
  pp::SimThread sim{&pp::HandTracker::now};
  pp::HandTracker tracker;
  const auto on_observation = [&sim](const pp::HandObservation& obs) { sim.push_hand(obs); };
  const auto set_camera = [&](bool on) {
    camera_wanted = on;
    if (on) tracker.start(on_observation);
    else tracker.stop();
    sim.send({.kind = pp::SimCommand::Kind::UseHand, .flag = on});
  };
  if (camera_wanted) set_camera(true);

  // Play data for later tuning (see src/game/telemetry.hpp for what and why).
  pp::SessionLog session;
  if (record) session.open(pp::SessionLog::default_root());
  double last_flush = 0;

  pp::PreviewImage preview;
  std::uint64_t preview_seq = 0;
  Texture2D preview_texture{};
  double camera_fps = 0, last_capture = 0, capture_to_display_ms = 0;
  std::uint64_t hands_seen = 0;

  Trail trail;
  real speed = 0.70f;
  bool assist = true, magnet = true;
  std::uint32_t paddle_hits = 0, table_hits = 0, net_hits = 0, player_hits = 0;
  float hit_flash = 0, hit_power = 0;
  std::uint64_t last_tick = 0;
  float sim_rate = 0;
  Vec3 pointer = pp::on_surface(0, 1.0f);

  std::array<float, 240> frame_ms{};
  std::size_t head = 0;

  while (!WindowShouldClose()) {
    PP_INTERVAL("frame");
    const float frame = std::min(GetFrameTime(), kMaxFrame);
    frame_ms[head] = frame * 1000.0f;
    head = (head + 1) % frame_ms.size();

    // ---- input: turn it into commands for the simulation ----
    using Kind = pp::SimCommand::Kind;
    if (IsKeyPressed(KEY_R)) sim.send({.kind = Kind::Restart});
    if (IsKeyPressed(KEY_A)) {
      assist = !assist;
      sim.send({.kind = Kind::SetAssist, .flag = assist});
    }
    if (IsKeyPressed(KEY_LEFT_BRACKET) || IsKeyPressed(KEY_RIGHT_BRACKET)) {
      speed = std::clamp(speed + (IsKeyPressed(KEY_LEFT_BRACKET) ? -0.05f : 0.05f), 0.40f, 1.00f);
      sim.send({.kind = Kind::SetSpeed, .value = speed});
    }
    if (IsKeyPressed(KEY_C)) set_camera(!camera_wanted);
    if (IsKeyPressed(KEY_K)) sim.send({.kind = Kind::Recentre});
    if (IsKeyPressed(KEY_M)) {
      magnet = !magnet;
      sim.send({.kind = Kind::SetMagnet, .flag = magnet});
    }
    pointer = mouse_on_surface(camera, pointer);
    sim.send({.kind = Kind::PointerTarget, .target = pointer});

    // ---- the newest state of the world ----
    // A copy: one consistent picture for the whole frame.
    const pp::Snapshot snap = sim.latest();
    trail.push(snap.ball.pos);

    if (snap.paddle_hits != paddle_hits) PlaySound(paddle_sound);
    if (snap.table_hits != table_hits) PlaySound(table_sound);
    if (snap.net_hits != net_hits) PlaySound(net_sound);
    paddle_hits = snap.paddle_hits;
    table_hits = snap.table_hits;
    net_hits = snap.net_hits;

    // Drain the simulation's event queue into the session file. Disk writes
    // belong here, on the thread that can afford a hiccup.
    pp::TelemetryEvent event;
    while (sim.poll_telemetry(event)) session.write(event);
    if (GetTime() - last_flush > 2.0) {
      session.flush();
      last_flush = GetTime();
    }

    if (frame > 0) {
      const float rate = static_cast<float>(snap.tick - last_tick) / frame;
      sim_rate += (rate - sim_rate) * 0.05f;
    }
    last_tick = snap.tick;

    // ---- camera bookkeeping ----
    if (camera_wanted) {
      if (snap.hands_received != hands_seen && last_capture > 0) {
        // Frame rate actually delivered, from the camera's own timestamps.
        const double gap = snap.hand.capture_time - last_capture;
        if (gap > 0) {
          camera_fps +=
              (static_cast<double>(snap.hands_received - hands_seen) / gap - camera_fps) * 0.1;
        }
      }
      hands_seen = snap.hands_received;
      last_capture = snap.hand.capture_time;

      if (tracker.copy_preview(preview, preview_seq)) {
        if (preview_texture.id == 0 || preview_texture.width != preview.width ||
            preview_texture.height != preview.height) {
          if (preview_texture.id != 0) UnloadTexture(preview_texture);
          const Image image{.data = preview.rgba.data(),
                            .width = preview.width,
                            .height = preview.height,
                            .mipmaps = 1,
                            .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
          preview_texture = LoadTextureFromImage(image);
        } else {
          UpdateTexture(preview_texture, preview.rgba.data());
        }
      }
    }

    // ---- draw ----
    const pp::Paddle& me = snap.paddles[pp::index(pp::Side::Player)];
    BeginDrawing();
    ClearBackground({16, 18, 24, 255});
    BeginMode3D(camera);
    DrawPlane({0, 0, 0}, {10, 10}, {30, 33, 42, 255});
    draw_table(pp::Table{}, pp::Net{});
    draw_bat(snap.paddles[pp::index(pp::Side::Opponent)], {52, 110, 190, 255}, {28, 28, 32, 255},
             1.0f);
    trail.draw(snap.ball_radius);
    draw_ball(snap.ball.pos, snap.ball_radius);
    if (snap.has_ring) {
      const Color ring = snap.ring_good ? Color{96, 196, 150, 255} : Color{214, 92, 80, 255};
      draw_ring(snap.ring, me.radius, ring);
      // Timing cue: an outer circle closes in on the ring as the ball
      // approaches. Swing as the two meet.
      if (snap.ring_good && snap.ball_due_in >= 0) {
        const float closing = std::clamp(snap.ball_due_in / 0.7f, 0.0f, 1.0f);
        draw_ring(snap.ring, me.radius * (1.0f + 2.4f * closing), Fade(RAYWHITE, 0.75f));
      }
    }
    // The bat lunges forward and back over the life of a swing, so a swing
    // looks like one.
    pp::Paddle shown = me;
    if (snap.swinging) {
      const float phase = std::clamp(snap.swing_age / snap.swing_live_for, 0.0f, 1.0f);
      shown.pos.z -= 0.22f * std::sin(std::numbers::pi_v<float> * std::sqrt(phase));
      shown.pos.y += 0.05f * std::sin(std::numbers::pi_v<float> * phase);
    }
    // Red on the side you see. A little see-through, so the ball stays visible.
    draw_bat(shown, {28, 28, 32, 255}, {206, 58, 52, 255}, snap.swinging ? 1.0f : 0.80f,
             snap.hand_in_control, snap.paddle_roll);
    EndMode3D();

    draw_hud(snap, session.is_open());
    // A struck ball says so, with how hard: the swing is the player's act and
    // should be acknowledged.
    if (snap.player_hits != player_hits) {
      player_hits = snap.player_hits;
      hit_flash = 0.6f;
      hit_power = snap.swing_power;
    }
    if (hit_flash > 0) {
      hit_flash -= frame;
      // The stroke the swing was read as: up is topspin, down is a chop.
      const char* word = snap.last_stroke == pp::StrokeKind::Topspin
                             ? (hit_power > 0.66f ? "TOPSPIN LOOP" : "TOPSPIN")
                         : snap.last_stroke == pp::StrokeKind::Chop ? "CHOP"
                         : hit_power > 0.66f ? "SMASH" : hit_power > 0.25f ? "DRIVE" : "TAP";
      draw_centered(word, GetScreenHeight() / 2 + 150, 30,
                    Fade({250, 176, 60, 255}, std::clamp(hit_flash / 0.3f, 0.0f, 1.0f)));
    }
    draw_frame_graph(frame_ms, head);
    draw_timing(snap, sim_rate, camera_wanted ? capture_to_display_ms : 0);
    if (camera_wanted) {
      const pp::CameraStatus status = tracker.status();
      const bool running = status == pp::CameraStatus::Running;
      const char* caption =
          running ? TextFormat("Camera %.0f fps   detect %.0f ms   dropped %llu", camera_fps,
                               1000.0 * (snap.hand.detect_time - snap.hand.capture_time),
                               static_cast<unsigned long long>(snap.hands_dropped))
                  : TextFormat("Camera: %s", pp::describe(status));
      const char* hint = !running ? ""
                         : snap.hand_in_control ? "Hand in control. K sets this spot as centre."
                                                : "Show an open palm to take control";
      draw_camera_overlay(preview_texture, snap.hand, snap.palm, caption, hint,
                          snap.hand_in_control ? Color{96, 196, 150, 255} : GRAY);
    }
    EndDrawing();  // swaps buffers: this frame is now on its way to the screen

    // End-to-end latency: from the camera exposing the frame to the picture
    // that used it being handed to the display. (The display adds its own
    // scan-out time on top, which software cannot see.)
    if (camera_wanted && snap.hand_in_control) {
      const double ms = 1000.0 * (pp::HandTracker::now() - snap.hand.capture_time);
      capture_to_display_ms += (ms - capture_to_display_ms) * 0.05;
    }

    if (screenshot_path != nullptr && ++frame_no == 300) {
      const Image shot = LoadImageFromScreen();
      ExportImage(shot, screenshot_path);
      UnloadImage(shot);
      break;
    }
  }

  tracker.stop();
  if (session.is_open()) {
    session.close();
    std::printf("session recorded in %s\n", session.directory().c_str());
  }
  if (preview_texture.id != 0) UnloadTexture(preview_texture);
  UnloadSound(paddle_sound);
  UnloadSound(table_sound);
  UnloadSound(net_sound);
  CloseAudioDevice();
  CloseWindow();
  return 0;
}
