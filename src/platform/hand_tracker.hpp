#pragma once
// Camera capture + hand pose detection.
//
// This header is plain C++: nothing here mentions Objective-C, AVFoundation
// or Vision. All of that lives in hand_tracker.mm behind the Impl pointer, so
// the rest of the codebase compiles as ordinary C++ and can be tested without
// a camera.
//
// Threading: observations are delivered on a background capture thread, one
// call per camera frame. The callback must be quick and must not touch
// anything the main thread uses without synchronisation.

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace pp {

// Order matches Vision's 21 hand landmarks.
enum class HandJoint : std::uint8_t {
  Wrist,
  ThumbCMC, ThumbMP, ThumbIP, ThumbTip,
  IndexMCP, IndexPIP, IndexDIP, IndexTip,
  MiddleMCP, MiddlePIP, MiddleDIP, MiddleTip,
  RingMCP, RingPIP, RingDIP, RingTip,
  LittleMCP, LittlePIP, LittleDIP, LittleTip,
  Count,
};
inline constexpr std::size_t kHandJointCount = static_cast<std::size_t>(HandJoint::Count);

// Normalised image coordinates: x right, y up, both in [0, 1]. The image is
// mirrored (like a mirror, not like a photo), so moving your hand to your
// right moves x toward 1.
struct JointPoint {
  float x = 0, y = 0;
  float confidence = 0;  // 0 = not seen
};

struct HandObservation {
  // Seconds on the host clock (see HandTracker::now), in pipeline order:
  //   capture_time  the camera exposed the frame
  //   receive_time  the frame reached this process
  //   detect_time   hand detection finished
  // detect - capture is the latency so far; the two gaps say whose it is.
  double capture_time = 0;
  double receive_time = 0;
  double detect_time = 0;
  bool found = false;    // false: a frame arrived but no hand was in it
  float aspect = 1.0f;   // image width / height, to undo the normalisation
  std::array<JointPoint, kHandJointCount> joints{};

  const JointPoint& operator[](HandJoint j) const { return joints[static_cast<std::size_t>(j)]; }
};

// A small mirrored copy of the camera image, for the debug overlay.
struct PreviewImage {
  int width = 0, height = 0;
  std::vector<std::uint8_t> rgba;  // width * height * 4
};

enum class CameraStatus : std::uint8_t {
  Stopped,
  WaitingForPermission,  // the macOS prompt is on screen
  Denied,                // refused; change it in System Settings > Privacy > Camera
  NoCamera,
  Running,
  Failed,
};
const char* describe(CameraStatus s);

struct CameraInfo {
  int width = 0, height = 0;
  double frames_per_second = 0;  // what the camera was configured for
};

class HandTracker {
 public:
  using Callback = std::function<void(const HandObservation&)>;

  HandTracker();
  ~HandTracker();  // stops the camera
  HandTracker(const HandTracker&) = delete;
  HandTracker& operator=(const HandTracker&) = delete;

  // Asks for camera permission if needed, then starts capturing. Returns at
  // once; watch status(). `on_observation` runs on the capture thread.
  void start(Callback on_observation);
  // Blocks until the capture thread has delivered its last callback.
  void stop();

  CameraStatus status() const;
  CameraInfo info() const;

  // Copies the newest preview frame into `out` if there is one newer than
  // `last_seen` (which is updated). Returns false when nothing changed.
  bool copy_preview(PreviewImage& out, std::uint64_t& last_seen) const;

  // The clock that observation timestamps are on.
  static double now();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pp
