#include "audio/sounds.hpp"

#include <raylib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace pp {

namespace {

constexpr int kRate = 44100;

// One struck-object sound: a few decaying tones (the object ringing) plus a
// very short burst of noise (the impact itself). The tones' frequencies and
// how fast they die away are what make a hollow celluloid ball on wood sound
// different from the same ball on rubber.
struct Partial {
  float frequency, level, decay;  // Hz, 0..1, 1/s
};

Sound synthesise(std::initializer_list<Partial> partials, float noise_level, float noise_decay,
                 float seconds) {
  const int frames = static_cast<int>(static_cast<float>(kRate) * seconds);
  auto* samples = static_cast<short*>(MemAlloc(static_cast<unsigned>(frames) * sizeof(short)));
  std::uint32_t rng = 0x2545F491u;
  float lowpassed = 0;
  for (int i = 0; i < frames; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(kRate);
    float v = 0;
    for (const Partial& p : partials) {
      v += p.level * std::exp(-t * p.decay) *
           std::sin(2.0f * std::numbers::pi_v<float> * p.frequency * t);
    }
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    const float white = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    lowpassed += (white - lowpassed) * 0.45f;
    v += noise_level * std::exp(-t * noise_decay) * lowpassed;
    const float attack = std::min(1.0f, t * 4000.0f);  // 0.25 ms: no click from a hard start
    samples[i] = static_cast<short>(std::clamp(v * attack, -1.0f, 1.0f) * 26000.0f);
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

// A few copies of each sound, so one can start while the last still rings.
struct Voice {
  std::array<Sound, 3> copies{};
  std::size_t next = 0;

  void load(Sound original) {
    copies[0] = original;
    for (std::size_t i = 1; i < copies.size(); ++i) copies[i] = LoadSoundAlias(original);
  }
  void unload() {
    for (std::size_t i = 1; i < copies.size(); ++i) UnloadSoundAlias(copies[i]);
    UnloadSound(copies[0]);
  }
  void play(float volume, float pitch, float side) {
    const Sound& s = copies[next];
    next = (next + 1) % copies.size();
    SetSoundVolume(s, std::clamp(volume, 0.0f, 1.0f));
    SetSoundPitch(s, std::clamp(pitch, 0.6f, 1.6f));
    // raylib: 0.5 is the centre and larger is further LEFT.
    SetSoundPan(s, std::clamp(0.5f - 0.30f * side, 0.0f, 1.0f));
    PlaySound(s);
  }
};

// Loudness of an impact at `speed`, relative to a hard one.
float loudness(float speed, float full_at) {
  return std::clamp(0.18f + 0.82f * speed / full_at, 0.0f, 1.0f);
}

}  // namespace

struct Sounds::Impl {
  Voice bat, table, net, floor;
};

Sounds::Sounds() : impl_(std::make_unique<Impl>()) {
  SetTraceLogLevel(LOG_WARNING);
  InitAudioDevice();
  // Bat: rubber over wood. Lower and rounder, a "pock".
  impl_->bat.load(synthesise({{760, 0.55f, 55}, {1480, 0.30f, 80}, {2900, 0.12f, 140}}, 0.55f, 420,
                             0.12f));
  // Table: a hard hollow ball on a hard board. Higher and much shorter, a "tick".
  impl_->table.load(synthesise({{1850, 0.50f, 95}, {3300, 0.32f, 150}, {5200, 0.12f, 240}}, 0.45f,
                               700, 0.08f));
  // Net: cloth. Nearly all noise, soft.
  impl_->net.load(synthesise({{210, 0.25f, 45}}, 0.50f, 60, 0.14f));
  // Floor: like the table but duller.
  impl_->floor.load(synthesise({{1100, 0.50f, 80}, {2300, 0.22f, 140}}, 0.40f, 500, 0.09f));
}

Sounds::~Sounds() {
  impl_->bat.unload();
  impl_->table.unload();
  impl_->net.unload();
  impl_->floor.unload();
  CloseAudioDevice();
}

void Sounds::bat(float speed, float side, float nearness) {
  const float v = loudness(speed, 11.0f) * (0.55f + 0.45f * nearness);
  impl_->bat.play(v, 0.92f + 0.022f * speed, side);
}

void Sounds::table(float speed, float side, float nearness) {
  const float v = loudness(speed, 9.0f) * (0.50f + 0.50f * nearness) * 0.85f;
  impl_->table.play(v, 0.94f + 0.018f * speed, side);
}

void Sounds::net(float speed) { impl_->net.play(loudness(speed, 8.0f) * 0.7f, 1.0f, 0); }

void Sounds::floor(float speed, float side) {
  impl_->floor.play(loudness(speed, 8.0f) * 0.5f, 0.95f, side);
}

}  // namespace pp
