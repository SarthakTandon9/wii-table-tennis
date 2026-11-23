// The Metal renderer. See metal_renderer.hpp for the design; this file is the
// part that talks to the GPU.
//
// Objective-C++: Metal objects are Objective-C objects (id<MTLDevice> and so
// on), reference-counted automatically by ARC. C++ containers hold the
// per-frame data. The two meet in Impl.
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

#include "render/metal_renderer.hpp"

#include "core/signpost.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <numbers>
#include <vector>

namespace pp::render {

namespace {

// ------------------------------------------------------------------ shaders
// Compiled from source when the renderer starts (a few milliseconds), so the
// build needs no offline Metal compiler.
NSString* const kShaderSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Frame {
  float4x4 view_proj;
  float4x4 light_view_proj;
  float4 eye;        // xyz
  float4 to_light;   // xyz: unit vector from a surface toward the lamp
  float4 table;      // half width, half length, surface height, net height
  float4 misc;       // x: 1 / shadow map size
};

struct Object {
  float4x4 model;
  float4 n0, n1, n2;  // columns of the normal matrix
  float4 color;       // linear rgb, alpha
  float4 material;    // x: material id
};

struct VIn {
  float3 position [[attribute(0)]];
  float3 normal [[attribute(1)]];
};

struct VOut {
  float4 position [[position]];
  float3 world;
  float3 normal;
  float4 in_light;  // position as the lamp sees it
};

vertex VOut scene_vs(VIn in [[stage_in]], constant Frame& frame [[buffer(1)]],
                     constant Object& object [[buffer(2)]]) {
  VOut out;
  const float4 world = object.model * float4(in.position, 1.0);
  out.world = world.xyz;
  out.normal = float3x3(object.n0.xyz, object.n1.xyz, object.n2.xyz) * in.normal;
  out.position = frame.view_proj * world;
  out.in_light = frame.light_view_proj * world;
  return out;
}

vertex float4 shadow_vs(VIn in [[stage_in]], constant Frame& frame [[buffer(1)]],
                        constant Object& object [[buffer(2)]]) {
  return frame.light_view_proj * (object.model * float4(in.position, 1.0));
}

// How much of the lamp reaches this point: 0 in shadow, 1 lit. Nine samples
// around the point soften the edge (percentage-closer filtering); each sample
// is itself a hardware-filtered comparison of four texels.
static float lamp_visibility(float4 in_light, float texel, depth2d<float> shadow_map,
                             sampler shadow_sampler) {
  const float3 p = in_light.xyz / in_light.w;
  const float2 uv = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || p.z > 1.0) return 1.0;
  float lit = 0.0;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      lit += shadow_map.sample_compare(shadow_sampler, uv + float2(x, y) * texel * 1.25, p.z - 0.0006);
    }
  }
  return lit / 9.0;
}

// Filmic tone curve (Narkowicz's fit to ACES): bright highlights roll off
// instead of clipping to flat white.
static float3 tone_map(float3 x) {
  return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fragment float4 scene_fs(VOut in [[stage_in]], constant Frame& frame [[buffer(1)]],
                         constant Object& object [[buffer(2)]],
                         depth2d<float> shadow_map [[texture(0)]],
                         sampler shadow_sampler [[sampler(0)]]) {
  const int material = int(object.material.x + 0.5);
  float3 base = object.color.rgb;
  float alpha = object.color.a;
  if (material == 7) return float4(base, alpha);  // glow: unlit

  float3 n = normalize(in.normal);
  const float3 to_eye = normalize(frame.eye.xyz - in.world);
  if (dot(n, to_eye) < 0.0) n = -n;  // lit from whichever side is looked at
  const float3 to_light = frame.to_light.xyz;

  float specular = 0.0, shininess = 16.0, emissive = 0.0, wrap = 0.0;
  if (material == 1) { specular = 0.35; shininess = 40.0; }
  if (material == 2) { specular = 0.10; shininess = 10.0; }
  if (material == 3) {
    // The table paints its own lines: 2 cm round the edge, 3 mm down the middle.
    specular = 0.45; shininess = 120.0;
    if (n.y > 0.9) {
      const float2 from_edge = frame.table.xy - abs(in.world.xz);
      const float edge = min(from_edge.x, from_edge.y);
      const float aa = fwidth(in.world.x) + fwidth(in.world.z);
      const float edge_line = 1.0 - smoothstep(0.020 - aa, 0.020 + aa, edge);
      const float thin = 0.5 * fwidth(in.world.x);
      const float centre_line = 1.0 - smoothstep(0.0015 - thin, 0.0015 + thin, abs(in.world.x));
      base = mix(base, float3(0.86, 0.88, 0.88), max(edge_line, centre_line));
    }
  }
  if (material == 4) {
    // Sports flooring: a faint large-scale mottle so it does not read as flat paint.
    const float2 g = in.world.xz * 1.7;
    const float mottle = sin(g.x * 3.1 + sin(g.y * 2.3)) * sin(g.y * 2.7 + sin(g.x * 1.9));
    base *= 0.94 + 0.06 * mottle;
    specular = 0.12; shininess = 30.0;
  }
  if (material == 5) {
    // Net: threads every 12 mm, a white tape along the top. Far away the
    // threads are thinner than a pixel, so the weave fades to its average.
    const float top = frame.table.z + frame.table.w;
    if (in.world.y > top - 0.015) {
      base = float3(0.90, 0.90, 0.88); alpha = 1.0;
    } else {
      const float2 cell = float2(in.world.x, in.world.y) / 0.012;
      const float2 d = abs(fract(cell) - 0.5);
      const float2 w = fwidth(cell);
      const float2 strand = smoothstep(0.5 - 0.16 - w, 0.5 - 0.16 + w, d);
      const float weave = max(strand.x, strand.y);
      const float coverage = mix(weave, 0.45, saturate(max(w.x, w.y) * 1.6));
      alpha = coverage * 0.92;
      base = float3(0.10, 0.11, 0.12);
    }
  }
  if (material == 6) { specular = 0.25; shininess = 24.0; emissive = 0.10; wrap = 0.35; }

  const float visible = lamp_visibility(in.in_light, frame.misc.x, shadow_map, shadow_sampler);
  // The hall is lit in a pool around the table and falls away into the dark.
  const float pool = 1.0 - 0.85 * smoothstep(2.6, 10.0, length(in.world.xz));

  const float n_dot_l = dot(n, to_light);
  const float diffuse = saturate((n_dot_l + wrap) / (1.0 + wrap));
  const float3 half_vector = normalize(to_light + to_eye);
  const float highlight = specular * pow(saturate(dot(n, half_vector)), shininess) *
                          step(0.0, n_dot_l);

  const float3 lamp = float3(1.00, 0.96, 0.90) * 1.9 * pool;
  const float3 sky = float3(0.42, 0.46, 0.55) * 0.45, ground = float3(0.30, 0.22, 0.18) * 0.30;
  const float3 ambient = mix(ground, sky, n.y * 0.5 + 0.5) * (0.45 + 0.55 * pool);

  float3 color = base * (ambient + lamp * diffuse * visible) + lamp * highlight * visible +
                 base * emissive;
  return float4(tone_map(color), alpha);
}

// ---- overlay ----
struct OverlayIn {
  float2 position [[attribute(0)]];
  float2 uv [[attribute(1)]];
  float4 color [[attribute(2)]];
};
struct OverlayOut {
  float4 position [[position]];
  float2 uv;
  float4 color;
};
struct OverlayUniforms {
  float2 view_size;
  float mode;  // 0: tinted by the glyph atlas, 1: an image
  float pad;
};

vertex OverlayOut overlay_vs(OverlayIn in [[stage_in]], constant OverlayUniforms& u [[buffer(1)]]) {
  OverlayOut out;
  out.position = float4(in.position.x / u.view_size.x * 2.0 - 1.0,
                        1.0 - in.position.y / u.view_size.y * 2.0, 0.0, 1.0);
  out.uv = in.uv;
  out.color = in.color;
  return out;
}

fragment float4 overlay_fs(OverlayOut in [[stage_in]], constant OverlayUniforms& u [[buffer(1)]],
                           texture2d<float> tex [[texture(0)]], sampler samp [[sampler(0)]]) {
  const float4 texel = tex.sample(samp, in.uv);
  if (u.mode > 0.5) return float4(texel.rgb, in.color.a);
  return float4(in.color.rgb, in.color.a * texel.r);
}
)METAL";

// ------------------------------------------------------------- shared structs
// Must match the shader structs byte for byte. float4x4 is 64 bytes and
// float4 is 16, with no padding between them.
struct FrameUniforms {
  Mat4 view_proj;
  Mat4 light_view_proj;
  float eye[4];
  float to_light[4];
  float table[4];
  float misc[4];
};

struct ObjectUniforms {
  Mat4 model;
  float n0[4], n1[4], n2[4];
  float color[4];
  float material[4];
};
// Each object's slice starts on a 256-byte boundary, the strictest alignment
// any Mac GPU asks for buffer offsets.
constexpr std::size_t kObjectStride = 256;
static_assert(sizeof(ObjectUniforms) <= kObjectStride);

struct OverlayVertex {
  float x, y, u, v;
  float r, g, b, a;
};

struct OverlayUniforms {
  float view_size[2];
  float mode;
  float pad;
};

struct Vertex {
  float px, py, pz, nx, ny, nz;
};

constexpr std::size_t kMaxObjects = 512;
constexpr std::size_t kMaxOverlayVertices = 96 * 1024;
constexpr int kFramesInFlight = 3;
constexpr int kShadowSize = 4096;

// sRGB to linear light. Lighting sums light, and light sums linearly; adding
// sRGB-encoded numbers gives results that are too dark in the midtones.
float to_linear(float c) {
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// ------------------------------------------------------------------ meshes
struct MeshData {
  std::vector<Vertex> vertices;
  std::vector<std::uint16_t> indices;
};

void add_quad(MeshData& m, Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 n) {
  const auto base = static_cast<std::uint16_t>(m.vertices.size());
  for (const Vec3 p : {a, b, c, d}) m.vertices.push_back({p.x, p.y, p.z, n.x, n.y, n.z});
  for (const int i : {0, 1, 2, 0, 2, 3}) m.indices.push_back(static_cast<std::uint16_t>(base + i));
}

MeshData make_box() {
  MeshData m;
  const float h = 0.5f;
  add_quad(m, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}, {0, 0, 1});
  add_quad(m, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {0, 0, -1});
  add_quad(m, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h}, {1, 0, 0});
  add_quad(m, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h}, {-1, 0, 0});
  add_quad(m, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h}, {0, 1, 0});
  add_quad(m, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h}, {0, -1, 0});
  return m;
}

MeshData make_sphere(int slices, int stacks) {
  MeshData m;
  const float pi = std::numbers::pi_v<float>;
  for (int j = 0; j <= stacks; ++j) {
    const float phi = pi * static_cast<float>(j) / static_cast<float>(stacks);
    for (int i = 0; i <= slices; ++i) {
      const float theta = 2 * pi * static_cast<float>(i) / static_cast<float>(slices);
      const Vec3 n{std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)};
      m.vertices.push_back({n.x, n.y, n.z, n.x, n.y, n.z});
    }
  }
  const int row = slices + 1;
  for (int j = 0; j < stacks; ++j) {
    for (int i = 0; i < slices; ++i) {
      const int a = j * row + i, b = a + row;
      for (const int k : {a, b, a + 1, a + 1, b, b + 1}) m.indices.push_back(static_cast<std::uint16_t>(k));
    }
  }
  return m;
}

MeshData make_cylinder(int segments) {
  MeshData m;
  const float pi = std::numbers::pi_v<float>;
  for (int i = 0; i <= segments; ++i) {
    const float theta = 2 * pi * static_cast<float>(i) / static_cast<float>(segments);
    const float c = std::cos(theta), s = std::sin(theta);
    m.vertices.push_back({c, 0, s, c, 0, s});
    m.vertices.push_back({c, 1, s, c, 0, s});
  }
  for (int i = 0; i < segments; ++i) {
    const int a = 2 * i;
    for (const int k : {a, a + 1, a + 2, a + 2, a + 1, a + 3}) m.indices.push_back(static_cast<std::uint16_t>(k));
  }
  for (const float y : {0.0f, 1.0f}) {  // caps: a fan around a centre vertex
    const float ny = y > 0.5f ? 1.0f : -1.0f;
    const auto centre = static_cast<std::uint16_t>(m.vertices.size());
    m.vertices.push_back({0, y, 0, 0, ny, 0});
    for (int i = 0; i <= segments; ++i) {
      const float theta = 2 * pi * static_cast<float>(i) / static_cast<float>(segments);
      m.vertices.push_back({std::cos(theta), y, std::sin(theta), 0, ny, 0});
    }
    for (int i = 0; i < segments; ++i) {
      m.indices.push_back(centre);
      m.indices.push_back(static_cast<std::uint16_t>(centre + 1 + i));
      m.indices.push_back(static_cast<std::uint16_t>(centre + 2 + i));
    }
  }
  return m;
}

MeshData make_ring(int segments, int sides, float tube) {
  MeshData m;
  const float pi = std::numbers::pi_v<float>;
  for (int i = 0; i <= segments; ++i) {
    const float theta = 2 * pi * static_cast<float>(i) / static_cast<float>(segments);
    const Vec3 out{std::cos(theta), std::sin(theta), 0};
    for (int j = 0; j <= sides; ++j) {
      const float phi = 2 * pi * static_cast<float>(j) / static_cast<float>(sides);
      const Vec3 n = out * std::cos(phi) + Vec3{0, 0, 1} * std::sin(phi);
      const Vec3 p = out + n * tube;
      m.vertices.push_back({p.x, p.y, p.z, n.x, n.y, n.z});
    }
  }
  const int row = sides + 1;
  for (int i = 0; i < segments; ++i) {
    for (int j = 0; j < sides; ++j) {
      const int a = i * row + j, b = a + row;
      for (const int k : {a, b, a + 1, a + 1, b, b + 1}) m.indices.push_back(static_cast<std::uint16_t>(k));
    }
  }
  return m;
}

struct GpuMesh {
  id<MTLBuffer> vertices;
  id<MTLBuffer> indices;
  NSUInteger index_count = 0;
};

// One queued 3D draw. The uniforms themselves are already in the frame's
// buffer; this is just which mesh and which slice.
struct DrawItem {
  Mesh mesh;
  bool casts_shadow;
  bool blended;
};

struct OverlayBatch {
  int mode;  // 0 atlas, 1 image
  std::uint32_t first, count;
};

// ---------------------------------------------------------------- the font
// ASCII glyphs drawn once with Core Text into a single-channel texture. Text
// is then two triangles per character, sampled from that atlas.
struct Glyph {
  float u0, v0, u1, v1;  // where in the atlas
  float advance;         // pen movement, in atlas pixels
};
constexpr int kAtlasW = 1024, kAtlasH = 512;
constexpr float kGlyphPixels = 56.0f;  // size the glyphs are drawn at
constexpr int kFirstChar = 32, kLastChar = 126;

}  // namespace

struct MetalRenderer::Impl {
  MTKView* view = nil;
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLRenderPipelineState> scene_pipeline, blend_pipeline, shadow_pipeline, overlay_pipeline;
  id<MTLDepthStencilState> depth_write, depth_read, depth_off;
  id<MTLSamplerState> shadow_sampler, linear_sampler;
  id<MTLTexture> shadow_map, atlas, preview;
  std::array<GpuMesh, static_cast<std::size_t>(Mesh::Count)> meshes;

  std::array<id<MTLBuffer>, kFramesInFlight> object_buffers, overlay_buffers;
  dispatch_semaphore_t in_flight = dispatch_semaphore_create(kFramesInFlight);
  int frame_index = 0;

  FrameUniforms frame{};
  CameraSetup camera;
  std::vector<DrawItem> items;
  std::size_t object_count = 0;
  std::vector<OverlayVertex> overlay;
  std::vector<OverlayBatch> batches;

  std::array<Glyph, kLastChar - kFirstChar + 1> glyphs{};
  float cell_height = 0, ascent = 0;
  float white_u = 0, white_v = 0;

  FrameStats stats;
  // Written by the GPU completion handler on another thread.
  std::atomic<float> gpu_ms{0};
  CFTimeInterval frame_began = 0;

  void build_pipelines();
  void build_meshes();
  void build_font();
  void push_overlay(int mode, const OverlayVertex* v, std::size_t n);
  void quad(int mode, float x0, float y0, float x1, float y1, float u0, float v0, float u1,
            float v1, Color c);
  void write_png(id<MTLTexture> texture, const char* path);
};

namespace {

id<MTLRenderPipelineState> make_pipeline(id<MTLDevice> device, MTLRenderPipelineDescriptor* d) {
  NSError* error = nil;
  id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:d error:&error];
  if (state == nil) NSLog(@"pipeline failed: %@", error);
  return state;
}

MTLVertexDescriptor* scene_layout() {
  MTLVertexDescriptor* v = [MTLVertexDescriptor vertexDescriptor];
  v.attributes[0].format = MTLVertexFormatFloat3;
  v.attributes[0].offset = 0;
  v.attributes[0].bufferIndex = 0;
  v.attributes[1].format = MTLVertexFormatFloat3;
  v.attributes[1].offset = 12;
  v.attributes[1].bufferIndex = 0;
  v.layouts[0].stride = sizeof(Vertex);
  return v;
}

void enable_blending(MTLRenderPipelineColorAttachmentDescriptor* c) {
  c.blendingEnabled = YES;
  c.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
  c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  c.sourceAlphaBlendFactor = MTLBlendFactorOne;
  c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
}

}  // namespace

void MetalRenderer::Impl::build_pipelines() {
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:kShaderSource options:nil error:&error];
  if (library == nil) {
    NSLog(@"shader compile failed: %@", error);
    return;
  }
  const NSUInteger samples = view.sampleCount;

  MTLRenderPipelineDescriptor* scene = [MTLRenderPipelineDescriptor new];
  scene.vertexFunction = [library newFunctionWithName:@"scene_vs"];
  scene.fragmentFunction = [library newFunctionWithName:@"scene_fs"];
  scene.vertexDescriptor = scene_layout();
  scene.colorAttachments[0].pixelFormat = view.colorPixelFormat;
  scene.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
  scene.rasterSampleCount = samples;
  scene_pipeline = make_pipeline(device, scene);
  enable_blending(scene.colorAttachments[0]);
  blend_pipeline = make_pipeline(device, scene);

  MTLRenderPipelineDescriptor* shadow = [MTLRenderPipelineDescriptor new];
  shadow.vertexFunction = [library newFunctionWithName:@"shadow_vs"];
  shadow.vertexDescriptor = scene_layout();
  shadow.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  shadow_pipeline = make_pipeline(device, shadow);

  MTLVertexDescriptor* layout = [MTLVertexDescriptor vertexDescriptor];
  layout.attributes[0].format = MTLVertexFormatFloat2;
  layout.attributes[0].offset = 0;
  layout.attributes[1].format = MTLVertexFormatFloat2;
  layout.attributes[1].offset = 8;
  layout.attributes[2].format = MTLVertexFormatFloat4;
  layout.attributes[2].offset = 16;
  layout.layouts[0].stride = sizeof(OverlayVertex);
  MTLRenderPipelineDescriptor* over = [MTLRenderPipelineDescriptor new];
  over.vertexFunction = [library newFunctionWithName:@"overlay_vs"];
  over.fragmentFunction = [library newFunctionWithName:@"overlay_fs"];
  over.vertexDescriptor = layout;
  over.colorAttachments[0].pixelFormat = view.colorPixelFormat;
  over.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
  over.rasterSampleCount = samples;
  enable_blending(over.colorAttachments[0]);
  overlay_pipeline = make_pipeline(device, over);

  MTLDepthStencilDescriptor* depth = [MTLDepthStencilDescriptor new];
  depth.depthCompareFunction = MTLCompareFunctionLess;
  depth.depthWriteEnabled = YES;
  depth_write = [device newDepthStencilStateWithDescriptor:depth];
  depth.depthWriteEnabled = NO;
  depth_read = [device newDepthStencilStateWithDescriptor:depth];
  depth.depthCompareFunction = MTLCompareFunctionAlways;
  depth_off = [device newDepthStencilStateWithDescriptor:depth];

  MTLSamplerDescriptor* compare = [MTLSamplerDescriptor new];
  compare.minFilter = MTLSamplerMinMagFilterLinear;
  compare.magFilter = MTLSamplerMinMagFilterLinear;
  compare.compareFunction = MTLCompareFunctionLess;
  shadow_sampler = [device newSamplerStateWithDescriptor:compare];
  MTLSamplerDescriptor* linear = [MTLSamplerDescriptor new];
  linear.minFilter = MTLSamplerMinMagFilterLinear;
  linear.magFilter = MTLSamplerMinMagFilterLinear;
  linear_sampler = [device newSamplerStateWithDescriptor:linear];

  MTLTextureDescriptor* map =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                         width:kShadowSize
                                                        height:kShadowSize
                                                     mipmapped:NO];
  map.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  map.storageMode = MTLStorageModePrivate;
  shadow_map = [device newTextureWithDescriptor:map];

  for (int i = 0; i < kFramesInFlight; ++i) {
    const auto slot = static_cast<std::size_t>(i);
    object_buffers[slot] = [device newBufferWithLength:kMaxObjects * kObjectStride
                                               options:MTLResourceStorageModeShared];
    overlay_buffers[slot] = [device newBufferWithLength:kMaxOverlayVertices * sizeof(OverlayVertex)
                                                options:MTLResourceStorageModeShared];
  }
}

void MetalRenderer::Impl::build_meshes() {
  const auto upload = [&](Mesh which, const MeshData& data) {
    GpuMesh& g = meshes[static_cast<std::size_t>(which)];
    g.vertices = [device newBufferWithBytes:data.vertices.data()
                                     length:data.vertices.size() * sizeof(Vertex)
                                    options:MTLResourceStorageModeShared];
    g.indices = [device newBufferWithBytes:data.indices.data()
                                    length:data.indices.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
    g.index_count = data.indices.size();
  };
  upload(Mesh::Box, make_box());
  upload(Mesh::Sphere, make_sphere(40, 24));
  upload(Mesh::Cylinder, make_cylinder(48));
  upload(Mesh::Ring, make_ring(96, 8, 0.030f));
}

void MetalRenderer::Impl::build_font() {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kAtlasW * kAtlasH), 0);
  CGColorSpaceRef gray = CGColorSpaceCreateDeviceGray();
  CGContextRef ctx = CGBitmapContextCreate(pixels.data(), kAtlasW, kAtlasH, 8, kAtlasW, gray,
                                           static_cast<CGBitmapInfo>(kCGImageAlphaNone));
  CGColorSpaceRelease(gray);
  CTFontRef font = CTFontCreateUIFontForLanguage(kCTFontUIFontEmphasizedSystem,
                                                 static_cast<CGFloat>(kGlyphPixels), nullptr);
  ascent = static_cast<float>(CTFontGetAscent(font));
  const float descent = static_cast<float>(CTFontGetDescent(font));
  const int pad = 3;
  cell_height = std::ceil(ascent + descent) + static_cast<float>(2 * pad);
  CGContextSetGrayFillColor(ctx, 1.0, 1.0);

  // A block of solid white in the corner: untextured shapes sample that, so
  // text and shapes share one pipeline and one draw call.
  CGContextFillRect(ctx, CGRectMake(0, kAtlasH - 8, 8, 8));
  white_u = 4.0f / static_cast<float>(kAtlasW);
  white_v = 4.0f / static_cast<float>(kAtlasH);

  float pen_x = 16, pen_y = 0;
  for (int c = kFirstChar; c <= kLastChar; ++c) {
    const UniChar ch = static_cast<UniChar>(c);
    CGGlyph glyph = 0;
    CTFontGetGlyphsForCharacters(font, &ch, &glyph, 1);
    CGSize advance{};
    CTFontGetAdvancesForGlyphs(font, kCTFontOrientationHorizontal, &glyph, &advance, 1);
    const float cell_width = std::ceil(static_cast<float>(advance.width)) + static_cast<float>(2 * pad);
    if (pen_x + cell_width > static_cast<float>(kAtlasW)) {
      pen_x = 0;
      pen_y += cell_height;
    }
    // Core Graphics counts y upward from the bottom; the texture counts down
    // from the top. The bitmap's first row is the top row.
    const CGPoint at = CGPointMake(static_cast<CGFloat>(pen_x + static_cast<float>(pad)),
                                   static_cast<CGFloat>(static_cast<float>(kAtlasH) - pen_y -
                                                        static_cast<float>(pad) - ascent));
    CTFontDrawGlyphs(font, &glyph, &at, 1, ctx);
    glyphs[static_cast<std::size_t>(c - kFirstChar)] = {
        pen_x / static_cast<float>(kAtlasW), pen_y / static_cast<float>(kAtlasH),
        (pen_x + cell_width) / static_cast<float>(kAtlasW),
        (pen_y + cell_height) / static_cast<float>(kAtlasH), static_cast<float>(advance.width)};
    pen_x += cell_width;
  }
  CFRelease(font);
  CGContextRelease(ctx);

  MTLTextureDescriptor* d =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                         width:kAtlasW
                                                        height:kAtlasH
                                                     mipmapped:NO];
  atlas = [device newTextureWithDescriptor:d];
  [atlas replaceRegion:MTLRegionMake2D(0, 0, kAtlasW, kAtlasH)
           mipmapLevel:0
             withBytes:pixels.data()
           bytesPerRow:kAtlasW];
}

MetalRenderer::MetalRenderer(void* mtk_view) : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.view = (__bridge MTKView*)mtk_view;
  s.device = s.view.device != nil ? s.view.device : MTLCreateSystemDefaultDevice();
  s.view.device = s.device;
  // An sRGB target: the shader writes linear light and the GPU encodes it.
  s.view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
  s.view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
  s.view.sampleCount = 4;
  s.view.framebufferOnly = NO;  // so a finished frame can be read back for a screenshot
  // Two drawables instead of the default three: one on screen, one being
  // drawn. A third lets finished frames queue up behind the display, and each
  // queued frame is 8 ms of extra delay between the hand moving and the
  // picture showing it.
  if ([s.view.layer isKindOfClass:[CAMetalLayer class]]) {
    static_cast<CAMetalLayer*>(s.view.layer).maximumDrawableCount = 2;
  }
  s.view.clearColor = MTLClearColorMake(0.004, 0.005, 0.008, 1.0);
  s.queue = [s.device newCommandQueue];
  s.items.reserve(kMaxObjects);
  s.overlay.reserve(kMaxOverlayVertices);
  s.batches.reserve(64);
  s.build_pipelines();
  s.build_meshes();
  s.build_font();
}

MetalRenderer::~MetalRenderer() {
  // Let frames still on the GPU finish before the buffers they read go away.
  for (int i = 0; i < kFramesInFlight; ++i) {
    dispatch_semaphore_wait(impl_->in_flight, DISPATCH_TIME_FOREVER);
  }
  for (int i = 0; i < kFramesInFlight; ++i) dispatch_semaphore_signal(impl_->in_flight);
}

float MetalRenderer::width() const { return static_cast<float>(impl_->view.bounds.size.width); }
float MetalRenderer::height() const { return static_cast<float>(impl_->view.bounds.size.height); }

FrameStats MetalRenderer::stats() const {
  FrameStats out = impl_->stats;
  out.gpu_ms = impl_->gpu_ms.load(std::memory_order_relaxed);
  return out;
}

void MetalRenderer::begin_frame(const CameraSetup& camera) {
  Impl& s = *impl_;
  // Wait here, not in end_frame: the buffer about to be written must not
  // still be in use by a frame the GPU has not finished.
  dispatch_semaphore_wait(s.in_flight, DISPATCH_TIME_FOREVER);
  s.frame_began = CACurrentMediaTime();
  s.frame_index = (s.frame_index + 1) % kFramesInFlight;
  s.items.clear();
  s.object_count = 0;
  s.overlay.clear();
  s.batches.clear();
  s.camera = camera;

  const float aspect = width() / std::max(height(), 1.0f);
  const Mat4 view = look_at(camera.eye, camera.target, {0, 1, 0});
  const Mat4 proj = perspective(camera.fov_y_degrees * std::numbers::pi_v<float> / 180.0f, aspect,
                                0.05f, 60.0f);
  s.frame.view_proj = proj * view;

  // The lamp: high above, a little toward the player and to one side, so the
  // ball's shadow falls just beyond the ball as the player sees it.
  const Vec3 to_light = normalized(Vec3{-0.22f, 1.0f, 0.30f});
  const Mat4 light_view = look_at(to_light * 9.0f, {0, 0.4f, 0}, {0, 0, -1});
  s.frame.light_view_proj = orthographic(3.4f, 3.4f, 1.0f, 14.0f) * light_view;
  const float eye[4] = {camera.eye.x, camera.eye.y, camera.eye.z, 1};
  const float light[4] = {to_light.x, to_light.y, to_light.z, 0};
  std::memcpy(s.frame.eye, eye, sizeof eye);
  std::memcpy(s.frame.to_light, light, sizeof light);
  const float table[4] = {1.525f * 0.5f, 2.74f * 0.5f, 0.76f, 0.1525f};
  std::memcpy(s.frame.table, table, sizeof table);
  s.frame.misc[0] = 1.0f / static_cast<float>(kShadowSize);
}

void MetalRenderer::draw(Mesh mesh, const Mat4& model, Color color, Material material,
                         bool casts_shadow) {
  Impl& s = *impl_;
  if (s.object_count >= kMaxObjects) return;
  // Written straight into the frame's GPU-visible buffer: no intermediate copy.
  auto* slot = reinterpret_cast<ObjectUniforms*>(
      static_cast<std::uint8_t*>(s.object_buffers[static_cast<std::size_t>(s.frame_index)].contents) +
      s.object_count * kObjectStride);
  slot->model = model;
  // Normals must be transformed by the inverse transpose of the model's 3x3,
  // or a squashed sphere would be lit as if it were round. For a matrix with
  // columns a, b, c that is [b x c, c x a, a x b] / det, and the division
  // only changes the length, which the shader normalises away.
  const Vec3 a{model.m[0][0], model.m[0][1], model.m[0][2]};
  const Vec3 b{model.m[1][0], model.m[1][1], model.m[1][2]};
  const Vec3 c{model.m[2][0], model.m[2][1], model.m[2][2]};
  const Vec3 n0 = cross(b, c), n1 = cross(c, a), n2 = cross(a, b);
  const float sign = dot(a, n0) < 0 ? -1.0f : 1.0f;
  const float cols[3][4] = {{n0.x * sign, n0.y * sign, n0.z * sign, 0},
                            {n1.x * sign, n1.y * sign, n1.z * sign, 0},
                            {n2.x * sign, n2.y * sign, n2.z * sign, 0}};
  std::memcpy(slot->n0, cols[0], sizeof cols[0]);
  std::memcpy(slot->n1, cols[1], sizeof cols[1]);
  std::memcpy(slot->n2, cols[2], sizeof cols[2]);
  const float linear[4] = {to_linear(color.r), to_linear(color.g), to_linear(color.b), color.a};
  std::memcpy(slot->color, linear, sizeof linear);
  slot->material[0] = static_cast<float>(material);

  const bool blended = color.a < 0.999f || material == Material::Net;
  s.items.push_back({mesh, casts_shadow && !blended, blended});
  ++s.object_count;
}

// ------------------------------------------------------------------ overlay

void MetalRenderer::Impl::push_overlay(int mode, const OverlayVertex* v, std::size_t n) {
  if (overlay.size() + n > kMaxOverlayVertices) return;
  const auto first = static_cast<std::uint32_t>(overlay.size());
  overlay.insert(overlay.end(), v, v + n);
  // Consecutive shapes that share a texture become one draw call.
  if (!batches.empty() && batches.back().mode == mode && mode == 0) {
    batches.back().count += static_cast<std::uint32_t>(n);
  } else {
    batches.push_back({mode, first, static_cast<std::uint32_t>(n)});
  }
}

void MetalRenderer::Impl::quad(int mode, float x0, float y0, float x1, float y1, float u0,
                               float v0, float u1, float v1, Color c) {
  const float r = to_linear(c.r), g = to_linear(c.g), b = to_linear(c.b);
  const OverlayVertex v[6] = {{x0, y0, u0, v0, r, g, b, c.a}, {x1, y0, u1, v0, r, g, b, c.a},
                              {x1, y1, u1, v1, r, g, b, c.a}, {x0, y0, u0, v0, r, g, b, c.a},
                              {x1, y1, u1, v1, r, g, b, c.a}, {x0, y1, u0, v1, r, g, b, c.a}};
  push_overlay(mode, v, 6);
}

void MetalRenderer::rect(float x, float y, float w, float h, Color color) {
  Impl& s = *impl_;
  s.quad(0, x, y, x + w, y + h, s.white_u, s.white_v, s.white_u, s.white_v, color);
}

void MetalRenderer::line(float x0, float y0, float x1, float y1, float width, Color color) {
  Impl& s = *impl_;
  const float dx = x1 - x0, dy = y1 - y0;
  const float len = std::hypot(dx, dy);
  if (len < 1e-4f) return;
  const float nx = -dy / len * width * 0.5f, ny = dx / len * width * 0.5f;
  const float r = to_linear(color.r), g = to_linear(color.g), b = to_linear(color.b);
  const float u = s.white_u, v = s.white_v;
  const OverlayVertex q[6] = {
      {x0 + nx, y0 + ny, u, v, r, g, b, color.a}, {x1 + nx, y1 + ny, u, v, r, g, b, color.a},
      {x1 - nx, y1 - ny, u, v, r, g, b, color.a}, {x0 + nx, y0 + ny, u, v, r, g, b, color.a},
      {x1 - nx, y1 - ny, u, v, r, g, b, color.a}, {x0 - nx, y0 - ny, u, v, r, g, b, color.a}};
  s.push_overlay(0, q, 6);
}

void MetalRenderer::circle(float x, float y, float radius, Color color) {
  Impl& s = *impl_;
  const int segments = std::clamp(static_cast<int>(radius * 1.5f), 12, 48);
  const float r = to_linear(color.r), g = to_linear(color.g), b = to_linear(color.b);
  const float u = s.white_u, v = s.white_v, pi = std::numbers::pi_v<float>;
  for (int i = 0; i < segments; ++i) {
    const float a0 = 2 * pi * static_cast<float>(i) / static_cast<float>(segments);
    const float a1 = 2 * pi * static_cast<float>(i + 1) / static_cast<float>(segments);
    const OverlayVertex tri[3] = {
        {x, y, u, v, r, g, b, color.a},
        {x + radius * std::cos(a0), y + radius * std::sin(a0), u, v, r, g, b, color.a},
        {x + radius * std::cos(a1), y + radius * std::sin(a1), u, v, r, g, b, color.a}};
    s.push_overlay(0, tri, 3);
  }
}

void MetalRenderer::circle_outline(float x, float y, float radius, float width, Color color) {
  const int segments = std::clamp(static_cast<int>(radius * 1.5f), 16, 64);
  const float pi = std::numbers::pi_v<float>;
  for (int i = 0; i < segments; ++i) {
    const float a0 = 2 * pi * static_cast<float>(i) / static_cast<float>(segments);
    const float a1 = 2 * pi * static_cast<float>(i + 1) / static_cast<float>(segments);
    line(x + radius * std::cos(a0), y + radius * std::sin(a0), x + radius * std::cos(a1),
         y + radius * std::sin(a1), width, color);
  }
}

float MetalRenderer::text_width(float size, std::string_view text) const {
  const float scale = size / kGlyphPixels;
  float w = 0;
  for (const char ch : text) {
    const int c = ch < kFirstChar || ch > kLastChar ? '?' : ch;
    w += impl_->glyphs[static_cast<std::size_t>(c - kFirstChar)].advance * scale;
  }
  return w;
}

void MetalRenderer::text(float x, float y, float size, Color color, std::string_view text) {
  Impl& s = *impl_;
  const float scale = size / kGlyphPixels;
  const float pad = 3.0f * scale;
  float pen = x;
  for (const char ch : text) {
    const int c = ch < kFirstChar || ch > kLastChar ? '?' : ch;
    const Glyph& g = s.glyphs[static_cast<std::size_t>(c - kFirstChar)];
    const float w = (g.u1 - g.u0) * static_cast<float>(kAtlasW) * scale;
    const float h = s.cell_height * scale;
    if (c != ' ') s.quad(0, pen - pad, y - pad, pen - pad + w, y - pad + h, g.u0, g.v0, g.u1, g.v1, color);
    pen += g.advance * scale;
  }
}

void MetalRenderer::image(const std::uint8_t* rgba, int w, int h, float x, float y, float dw,
                          float dh) {
  Impl& s = *impl_;
  if (rgba == nullptr || w <= 0 || h <= 0) return;
  if (s.preview == nil || static_cast<int>(s.preview.width) != w ||
      static_cast<int>(s.preview.height) != h) {
    MTLTextureDescriptor* d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                                           width:static_cast<NSUInteger>(w)
                                                          height:static_cast<NSUInteger>(h)
                                                       mipmapped:NO];
    s.preview = [s.device newTextureWithDescriptor:d];
  }
  [s.preview replaceRegion:MTLRegionMake2D(0, 0, static_cast<NSUInteger>(w), static_cast<NSUInteger>(h))
               mipmapLevel:0
                 withBytes:rgba
               bytesPerRow:static_cast<NSUInteger>(w) * 4];
  s.quad(1, x, y, x + dw, y + dh, 0, 0, 1, 1, {1, 1, 1, 1});
}

void MetalRenderer::ray_through(float x, float y, Vec3& origin, Vec3& direction) const {
  const CameraSetup& cam = impl_->camera;
  const Vec3 forward = normalized(cam.target - cam.eye);
  const Vec3 right = normalized(cross(forward, {0, 1, 0}));
  const Vec3 up = cross(right, forward);
  const float tan_half = std::tan(cam.fov_y_degrees * std::numbers::pi_v<float> / 360.0f);
  const float aspect = width() / std::max(height(), 1.0f);
  const float nx = (x / std::max(width(), 1.0f) * 2.0f - 1.0f) * tan_half * aspect;
  const float ny = (1.0f - y / std::max(height(), 1.0f) * 2.0f) * tan_half;
  origin = cam.eye;
  direction = normalized(forward + right * nx + up * ny);
}

// -------------------------------------------------------------- end of frame

void MetalRenderer::Impl::write_png(id<MTLTexture> texture, const char* path) {
  const NSUInteger w = texture.width, h = texture.height;
  std::vector<std::uint8_t> pixels(w * h * 4);
  [texture getBytes:pixels.data()
        bytesPerRow:w * 4
         fromRegion:MTLRegionMake2D(0, 0, w, h)
        mipmapLevel:0];
  CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGContextRef ctx = CGBitmapContextCreate(
      pixels.data(), w, h, 8, w * 4, srgb,
      static_cast<CGBitmapInfo>(kCGImageAlphaNoneSkipFirst) | kCGBitmapByteOrder32Little);
  CGImageRef image = CGBitmapContextCreateImage(ctx);
  NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  CGImageDestinationRef dest =
      CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
  if (dest != nullptr) {
    CGImageDestinationAddImage(dest, image, nullptr);
    CGImageDestinationFinalize(dest);
    CFRelease(dest);
  }
  CGImageRelease(image);
  CGContextRelease(ctx);
  CGColorSpaceRelease(srgb);
}

void MetalRenderer::end_frame(const char* png_path) {
  Impl& s = *impl_;
  PP_INTERVAL("encode");
  const auto slot = static_cast<std::size_t>(s.frame_index);
  id<MTLBuffer> objects = s.object_buffers[slot];
  id<MTLBuffer> overlay_buffer = s.overlay_buffers[slot];
  if (!s.overlay.empty()) {
    std::memcpy(overlay_buffer.contents, s.overlay.data(), s.overlay.size() * sizeof(OverlayVertex));
  }

  id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
  int draws = 0;
  const auto draw_items = [&](id<MTLRenderCommandEncoder> enc, bool shadow_pass, bool blended) {
    Mesh bound = Mesh::Count;
    for (std::size_t i = 0; i < s.items.size(); ++i) {
      const DrawItem& item = s.items[i];
      if (shadow_pass ? !item.casts_shadow : item.blended != blended) continue;
      const GpuMesh& mesh = s.meshes[static_cast<std::size_t>(item.mesh)];
      if (item.mesh != bound) {  // skip redundant state changes
        [enc setVertexBuffer:mesh.vertices offset:0 atIndex:0];
        bound = item.mesh;
      }
      // Same buffer every time; only the offset moves. Cheaper than binding.
      [enc setVertexBufferOffset:i * kObjectStride atIndex:2];
      if (!shadow_pass) [enc setFragmentBufferOffset:i * kObjectStride atIndex:2];
      [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                      indexCount:mesh.index_count
                       indexType:MTLIndexTypeUInt16
                     indexBuffer:mesh.indices
               indexBufferOffset:0];
      ++draws;
    }
  };

  // ---- pass 1: depth from the lamp ----
  {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.depthAttachment.texture = s.shadow_map;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = 1.0;
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label = @"shadow";
    [enc setRenderPipelineState:s.shadow_pipeline];
    [enc setDepthStencilState:s.depth_write];
    // Push the stored depth away from the lamp a little, more on surfaces
    // seen edge-on, so a surface does not shadow itself ("shadow acne").
    [enc setDepthBias:2.0f slopeScale:2.5f clamp:0.02f];
    [enc setVertexBytes:&s.frame length:sizeof s.frame atIndex:1];
    [enc setVertexBuffer:objects offset:0 atIndex:2];
    draw_items(enc, true, false);
    [enc endEncoding];
  }

  // ---- pass 2: the scene, then the overlay ----
  // Asking for the drawable can block until the display is ready for another
  // frame. That wait is pacing, not work, so it is kept out of the CPU time.
  const CFTimeInterval wait_began = CACurrentMediaTime();
  MTLRenderPassDescriptor* pass = s.view.currentRenderPassDescriptor;
  id<CAMetalDrawable> drawable = s.view.currentDrawable;
  const CFTimeInterval waited = CACurrentMediaTime() - wait_began;
  if (pass != nil && drawable != nil) {
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label = @"scene";
    [enc setVertexBytes:&s.frame length:sizeof s.frame atIndex:1];
    [enc setFragmentBytes:&s.frame length:sizeof s.frame atIndex:1];
    [enc setVertexBuffer:objects offset:0 atIndex:2];
    [enc setFragmentBuffer:objects offset:0 atIndex:2];
    [enc setFragmentTexture:s.shadow_map atIndex:0];
    [enc setFragmentSamplerState:s.shadow_sampler atIndex:0];

    [enc setRenderPipelineState:s.scene_pipeline];
    [enc setDepthStencilState:s.depth_write];
    draw_items(enc, false, false);
    // See-through things last, without writing depth, so what is behind
    // them has already been drawn and still shows.
    [enc setRenderPipelineState:s.blend_pipeline];
    [enc setDepthStencilState:s.depth_read];
    draw_items(enc, false, true);

    if (!s.batches.empty()) {
      [enc setRenderPipelineState:s.overlay_pipeline];
      [enc setDepthStencilState:s.depth_off];
      [enc setVertexBuffer:overlay_buffer offset:0 atIndex:0];
      [enc setFragmentSamplerState:s.linear_sampler atIndex:0];
      for (const OverlayBatch& batch : s.batches) {
        const OverlayUniforms u{{width(), height()}, static_cast<float>(batch.mode), 0};
        [enc setVertexBytes:&u length:sizeof u atIndex:1];
        [enc setFragmentBytes:&u length:sizeof u atIndex:1];
        [enc setFragmentTexture:(batch.mode == 1 && s.preview != nil ? s.preview : s.atlas) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:batch.first vertexCount:batch.count];
        ++draws;
      }
    }
    [enc endEncoding];
    [cmd presentDrawable:drawable];
  }

  // Runs on a Metal thread when the GPU has finished this frame.
  Impl* impl = impl_.get();
  dispatch_semaphore_t in_flight = s.in_flight;
  [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
    const float ms = static_cast<float>((done.GPUEndTime - done.GPUStartTime) * 1000.0);
    const float old = impl->gpu_ms.load(std::memory_order_relaxed);
    impl->gpu_ms.store(old + (ms - old) * 0.1f, std::memory_order_relaxed);
    dispatch_semaphore_signal(in_flight);
  }];
  [cmd commit];

  const float cpu = static_cast<float>((CACurrentMediaTime() - s.frame_began - waited) * 1000.0);
  s.stats.cpu_ms += (cpu - s.stats.cpu_ms) * 0.1f;
  s.stats.draws = draws;

  if (png_path != nullptr && drawable != nil) {
    [cmd waitUntilCompleted];
    s.write_png(drawable.texture, png_path);
  }
}

}  // namespace pp::render
