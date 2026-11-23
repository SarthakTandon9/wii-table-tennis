#include "game/session_log.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace pp {

namespace {

// The record size is part of the header: if TelemetryEvent ever changes
// shape, old files are rejected instead of being misread.
struct Header {
  char magic[8] = {'P', 'P', 'S', 'E', 'S', 'S', '0', '2'};
  std::uint32_t record_size = sizeof(TelemetryEvent);
  std::uint32_t reserved = 0;
};

constexpr const char* kFileName = "events.bin";

}  // namespace

std::filesystem::path SessionLog::default_root() {
  const char* home = std::getenv("HOME");
  return std::filesystem::path{home != nullptr ? home : "."} / "Library" / "Application Support" /
         "PalmRally" / "sessions";
}

bool SessionLog::open(const std::filesystem::path& root) {
  close();
  char stamp[32];
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &local);

  std::error_code ec;
  dir_ = root / stamp;
  std::filesystem::create_directories(dir_, ec);
  if (ec) return false;

  file_ = std::fopen((dir_ / kFileName).c_str(), "wb");
  if (file_ == nullptr) return false;
  const Header header;
  std::fwrite(&header, sizeof header, 1, file_);
  return true;
}

void SessionLog::write(const TelemetryEvent& event) {
  if (file_ != nullptr) std::fwrite(&event, sizeof event, 1, file_);
}

void SessionLog::flush() {
  if (file_ != nullptr) std::fflush(file_);
}

void SessionLog::close() {
  if (file_ != nullptr) std::fclose(file_);
  file_ = nullptr;
}

std::optional<SessionData> load_session(const std::filesystem::path& dir) {
  std::FILE* file = std::fopen((dir / kFileName).c_str(), "rb");
  if (file == nullptr) return std::nullopt;

  Header header{}, expected;
  const bool ok = std::fread(&header, sizeof header, 1, file) == 1 &&
                  std::memcmp(header.magic, expected.magic, sizeof header.magic) == 0 &&
                  header.record_size == expected.record_size;
  if (!ok) {
    std::fclose(file);
    return std::nullopt;
  }

  SessionData data;
  TelemetryEvent event;
  while (std::fread(&event, sizeof event, 1, file) == 1) {
    switch (event.kind) {
      case TelemetryEvent::Kind::Hand: data.hands.push_back(event.hand); break;
      case TelemetryEvent::Kind::Chance: data.chances.push_back(event.chance); break;
      case TelemetryEvent::Kind::Point: data.points.push_back(event.point); break;
      case TelemetryEvent::Kind::Swing: data.swings.push_back(event.swing); break;
    }
  }
  std::fclose(file);
  return data;
}

std::optional<std::filesystem::path> latest_session(const std::filesystem::path& root) {
  std::error_code ec;
  std::optional<std::filesystem::path> newest;
  for (const auto& entry : std::filesystem::directory_iterator{root, ec}) {
    if (!entry.is_directory()) continue;
    // Names are date-time stamps, so the greatest name is the newest.
    if (!newest || entry.path().filename() > newest->filename()) newest = entry.path();
  }
  return newest;
}

}  // namespace pp
