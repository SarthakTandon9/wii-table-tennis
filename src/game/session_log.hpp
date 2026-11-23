#pragma once
// Recording a play session to disk, and reading one back.
//
// One file per session: a short header, then the telemetry events exactly as
// they are in memory. Fixed-size binary records are the simplest format that
// is also exact (floating-point values survive untouched), and a session is
// small: about 15 KB per second with the camera on.
//
// Writing happens on the main thread, never on the simulation thread: a disk
// write can stall for milliseconds, which the renderer can absorb and a 1 kHz
// loop cannot.

#include "game/telemetry.hpp"

#include <cstdio>
#include <filesystem>
#include <optional>
#include <vector>

namespace pp {

class SessionLog {
 public:
  SessionLog() = default;
  ~SessionLog() { close(); }
  SessionLog(const SessionLog&) = delete;
  SessionLog& operator=(const SessionLog&) = delete;

  // ~/Library/Application Support/PalmRally/sessions
  static std::filesystem::path default_root();

  // Creates <root>/<date-time>/events.bin. Returns false if it cannot.
  bool open(const std::filesystem::path& root);
  void write(const TelemetryEvent& event);
  void flush();
  void close();

  bool is_open() const { return file_ != nullptr; }
  const std::filesystem::path& directory() const { return dir_; }

 private:
  std::FILE* file_ = nullptr;
  std::filesystem::path dir_;
};

struct SessionData {
  std::vector<HandObservation> hands;
  std::vector<ChanceRecord> chances;
  std::vector<PointRecord> points;
  std::vector<SwingRecord> swings;
};

// `dir` is one session's directory. Returns nothing if the file is missing or
// was written by an incompatible build.
std::optional<SessionData> load_session(const std::filesystem::path& dir);

// The most recent session under `root`, if any.
std::optional<std::filesystem::path> latest_session(const std::filesystem::path& root);

}  // namespace pp
