#pragma once
// The game as the player sees and hears it: input in, picture and sound out.
// Plain C++. The window and its events belong to metal_main.mm, the GPU to
// MetalRenderer; this sits between them and the simulation.

#include "render/metal_renderer.hpp"

#include <memory>

namespace pp {

struct AppOptions {
  bool camera = false;               // start with the camera on
  bool record = true;                // write a session log
  const char* screenshot = nullptr;  // render for a while, save one frame here, quit
  int screenshot_frame = 300;        // which frame that is
  int bench_frames = 0;              // run this many frames, print the renderer's timings, quit
  bool demo = false;                 // move the mouse bat by itself (for unattended checks)
};

class GameApp {
 public:
  GameApp(render::MetalRenderer& renderer, const AppOptions& options);
  ~GameApp();

  void key_pressed(char key);        // lower-case ASCII; '[' and ']' included
  void mouse_moved(float x, float y);  // points, from the top-left of the view

  // One displayed frame. `dt` is the time since the previous one, in seconds.
  void frame(float dt);

  bool wants_quit() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pp
