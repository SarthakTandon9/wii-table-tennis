// Headless check of the camera pipeline: runs the tracker for a few seconds
// and reports frame rate, detection latency and the palm position.
//
//   hand_probe [seconds]
//
// Note that it is ordinary C++. It includes only the C++ header and never
// sees an Objective-C type.
#include "estimation/palm.hpp"
#include "platform/hand_tracker.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

int main(int argc, char** argv) {
  const double run_seconds = argc > 1 ? std::atof(argv[1]) : 5.0;

  struct Stats {
    std::mutex mutex;
    int frames = 0, hands = 0;
    double latency_sum = 0, latency_max = 0, camera_sum = 0, vision_sum = 0;
    double first_capture = 0, last_capture = 0, warmup_until = 0;
    pp::Palm palm;
    pp::HandObservation last_hand;
  } stats;

  pp::HandTracker tracker;
  tracker.start([&](const pp::HandObservation& obs) {
    const double latency = obs.detect_time - obs.capture_time;
    std::lock_guard lock{stats.mutex};
    // The first detection loads Vision's model and takes about a second.
    // Leave the warm-up out so the numbers describe steady running.
    if (stats.warmup_until == 0) stats.warmup_until = obs.detect_time + 1.0;
    if (obs.detect_time < stats.warmup_until) return;
    if (stats.frames++ == 0) stats.first_capture = obs.capture_time;
    stats.last_capture = obs.capture_time;
    stats.latency_sum += latency;
    stats.camera_sum += obs.receive_time - obs.capture_time;
    stats.vision_sum += obs.detect_time - obs.receive_time;
    stats.latency_max = std::max(stats.latency_max, latency);
    stats.palm = pp::palm_from(obs);
    if (obs.found) {
      ++stats.hands;
      stats.last_hand = obs;
    }
  });

  // start() returns at once. Give the camera a moment to come up, and the
  // user up to 30 s to answer the permission prompt if macOS shows one.
  using clock = std::chrono::steady_clock;
  const auto began = clock::now();
  auto waited = [&] { return std::chrono::duration<double>(clock::now() - began).count(); };
  for (;;) {
    const pp::CameraStatus s = tracker.status();
    const bool starting = s == pp::CameraStatus::Stopped && waited() < 3.0;
    const bool prompting = s == pp::CameraStatus::WaitingForPermission && waited() < 30.0;
    if (!starting && !prompting) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::printf("camera: %s\n", pp::describe(tracker.status()));
  if (tracker.status() != pp::CameraStatus::Running) return 1;

  const auto end = clock::now() + std::chrono::duration<double>(run_seconds);
  while (clock::now() < end) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::lock_guard lock{stats.mutex};
    if (stats.palm.valid) {
      std::printf("  frames %4d  hands %4d  palm (%.2f, %.2f) size %.3f roll %+.0f deg conf %.2f\n",
                  stats.frames, stats.hands, static_cast<double>(stats.palm.x),
                  static_cast<double>(stats.palm.y), static_cast<double>(stats.palm.size),
                  static_cast<double>(stats.palm.roll) * 57.2958,
                  static_cast<double>(stats.palm.confidence));
    } else {
      std::printf("  frames %4d  hands %4d  (no usable palm; weakest landmark conf %.2f)\n",
                  stats.frames, stats.hands, static_cast<double>(stats.palm.confidence));
    }
  }
  tracker.stop();

  const pp::CameraInfo info = tracker.info();
  std::lock_guard lock{stats.mutex};
  if (stats.last_hand.found) {
    // Wrist first, then four joints per finger from thumb to little finger.
    std::printf("\nlast hand, landmark confidence and position:\n");
    for (std::size_t i = 0; i < pp::kHandJointCount; ++i) {
      const pp::JointPoint& j = stats.last_hand.joints[i];
      std::printf("%s%.2f(%.2f,%.2f)", i == 0 ? "  " : (i - 1) % 4 == 0 ? "\n  " : "  ",
                  static_cast<double>(j.confidence), static_cast<double>(j.x),
                  static_cast<double>(j.y));
    }
    std::printf("\n");
  }
  const double span = stats.last_capture - stats.first_capture;
  std::printf("\nimage            %d x %d, configured for %.0f fps\n", info.width, info.height,
              info.frames_per_second);
  if (stats.frames > 1 && span > 0) {
    std::printf("delivered        %d frames, %.1f per second\n", stats.frames,
                (stats.frames - 1) / span);
    std::printf("capture->detect  mean %.1f ms, worst %.1f ms\n",
                1000.0 * stats.latency_sum / stats.frames, 1000.0 * stats.latency_max);
    std::printf("  camera+driver  mean %.1f ms   (exposure to frame in hand)\n",
                1000.0 * stats.camera_sum / stats.frames);
    std::printf("  Vision         mean %.1f ms   (hand pose detection)\n",
                1000.0 * stats.vision_sum / stats.frames);
    std::printf("hand found in    %d of %d frames\n", stats.hands, stats.frames);
  }
  return 0;
}
