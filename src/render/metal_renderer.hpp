#pragma once
// The Metal renderer, seen from C++.
//
// Like the camera bridge, this header mentions nothing from Objective-C or
// Metal; that all lives in metal_renderer.mm behind the Impl pointer.
//
// How a frame is drawn:
//
//   1. Shadow pass. The scene is drawn from the lamp's point of view into a
//      depth-only texture. Each texel then holds "how far from the lamp is
//      the nearest thing in this direction".
//   2. Main pass, 4x multisampled. The scene is drawn from the eye. For each
//      pixel the shader asks the shadow texture whether something sits
//      between this point and the lamp, and lights the pixel accordingly.
//      That is what puts a real shadow under the ball, which is the cue the
//      eye uses to judge its height and distance.
//   3. Overlay. Text, lines and images, in pixels, alpha blended on top.
//
// What it costs, and why it is laid out the way it is:
//
//  - The caller queues draws; nothing reaches the GPU until end_frame().
//    Queued draws live in arrays that are reused every frame, so a frame
//    allocates nothing once the arrays have grown to their working size.
//  - Per-object data (matrix, colour, material) for the whole frame goes in
//    ONE buffer, written front to back, and each draw just points at its
//    slice. One upload per frame instead of one per object.
//  - That buffer is shared between CPU and GPU (Apple silicon has one memory
//    for both), so "upload" is a memcpy with no driver copy behind it.
//  - Three such buffers are used in rotation. The GPU may still be reading
//    the previous frame's while the CPU writes the next; a semaphore stops
//    the CPU getting more than three frames ahead.

#include "math/vec3.hpp"
#include "render/mat4.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace pp::render {

// Colours are given the way they are written in a paint program (sRGB,
// 0..1); the renderer converts to linear light for the lighting maths.
struct Color {
  float r = 1, g = 1, b = 1, a = 1;
};
constexpr Color rgb(int r, int g, int b, float a = 1.0f) {
  return {static_cast<float>(r) / 255.0f, static_cast<float>(g) / 255.0f,
          static_cast<float>(b) / 255.0f, a};
}
constexpr Color fade(Color c, float alpha) { return {c.r, c.g, c.b, c.a * alpha}; }

// Unit shapes, each built once. Scaled and placed by the model matrix.
enum class Mesh : std::uint8_t {
  Box,       // -0.5 .. 0.5 on each axis
  Sphere,    // radius 1
  Cylinder,  // radius 1 around y, from y = 0 to y = 1, capped
  Ring,      // a thin torus of radius 1 in the xy plane
  Count,
};

// How a surface answers light. Some are procedural: the table paints its own
// white lines and the net its own mesh, from the position being shaded.
enum class Material : std::uint8_t {
  Matte,     // cloth, skin, paint
  Gloss,     // lacquered wood, plastic: a soft highlight
  Rubber,    // bat rubber: dull, slightly grippy sheen
  Table,     // painted top with lines and a faint reflection of the lamp
  Floor,     // sports-hall floor, lit in a pool around the table
  Net,       // woven mesh, see-through between the threads
  Ball,      // celluloid: bright, slightly translucent look
  Glow,      // unlit: the on-screen aids (ring, timing circle)
};

struct CameraSetup {
  Vec3 eye{0, 1.55f, 3.25f};
  Vec3 target{0, 0.82f, 0};
  float fov_y_degrees = 45.0f;
};

struct FrameStats {
  float cpu_ms = 0;   // time spent building and encoding the frame
  float gpu_ms = 0;   // time the GPU spent executing it (previous frames, smoothed)
  int draws = 0;      // draw calls issued, all passes
};

class MetalRenderer {
 public:
  // `mtk_view` is an MTKView*. The renderer configures it (pixel formats,
  // multisampling) and draws into it.
  explicit MetalRenderer(void* mtk_view);
  ~MetalRenderer();
  MetalRenderer(const MetalRenderer&) = delete;
  MetalRenderer& operator=(const MetalRenderer&) = delete;

  // ---- one frame ----
  void begin_frame(const CameraSetup& camera);
  void draw(Mesh mesh, const Mat4& model, Color color, Material material,
            bool casts_shadow = true);

  // Overlay, in points from the top-left of the view.
  void rect(float x, float y, float w, float h, Color color);
  void line(float x0, float y0, float x1, float y1, float width, Color color);
  void circle(float x, float y, float radius, Color color);                 // filled
  void circle_outline(float x, float y, float radius, float width, Color color);
  void text(float x, float y, float size, Color color, std::string_view text);
  float text_width(float size, std::string_view text) const;
  // An RGBA image (the camera preview). The pixels are copied before this returns.
  void image(const std::uint8_t* rgba, int width, int height, float x, float y, float w, float h);

  // Encodes everything queued and presents it. With `png_path` set, also
  // writes the finished picture to that file (and blocks until it is done).
  void end_frame(const char* png_path = nullptr);

  float width() const;   // of the view, in points
  float height() const;
  FrameStats stats() const;

  // The ray through a point of the view (in points), for mouse picking.
  void ray_through(float x, float y, Vec3& origin, Vec3& direction) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pp::render
