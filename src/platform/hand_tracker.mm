// Objective-C++ implementation of HandTracker.
//
// A .mm file is compiled as C++ and Objective-C at once: C++ objects can hold
// Objective-C object pointers and the other way round. With ARC switched on
// (-fobjc-arc, set in CMake) those pointers are reference-counted for us, so
// an Objective-C pointer inside a C++ struct behaves like a shared_ptr: it is
// retained on assignment and released when the struct is destroyed.
//
// Pipeline, all on one serial dispatch queue ("the capture thread"):
//   camera frame -> Vision hand pose -> HandObservation -> user callback
// Running detection on the capture queue means a slow detection makes the
// camera drop frames (alwaysDiscardsLateVideoFrames) instead of queueing them
// up. Stale frames are worthless for a paddle; we always want the newest.

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

#include "core/signpost.hpp"
#include "platform/hand_tracker.hpp"

#include <atomic>
#include <mutex>

namespace pp {
struct HandTrackerShared;
}

// Receives frames from AVFoundation. Objective-C needs an object here; it
// simply forwards to the C++ side.
@interface PPFrameReceiver : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
- (instancetype)initWithShared:(std::shared_ptr<pp::HandTrackerShared>)shared;
@end

namespace pp {

namespace {

constexpr int kPreviewWidth = 320;

double seconds(CMTime t) { return CMTIME_IS_VALID(t) ? CMTimeGetSeconds(t) : 0.0; }

}  // namespace

// Everything the tracker owns. It is held by shared_ptr from three places:
// the HandTracker, the frame receiver, and any block waiting on a dispatch
// queue (for example the answer to the permission prompt, which can arrive
// long after the HandTracker itself is gone). Whoever lets go last frees it,
// so no thread can ever be left holding a dangling pointer.
struct HandTrackerShared : std::enable_shared_from_this<HandTrackerShared> {
  HandTracker::Callback callback;        // set before capture starts, then read-only
  std::atomic<CameraStatus> status{CameraStatus::Stopped};
  std::atomic<bool> wanted{false};       // start() was called and stop() has not been
  std::atomic<bool> accepting{false};    // false: drop frames (we are stopping)

  // Two serial queues. `control` runs configure/start/stop, so those never
  // overlap. `capture` is where frames arrive and detection runs.
  dispatch_queue_t control = dispatch_queue_create("palmrally.camera.control", DISPATCH_QUEUE_SERIAL);
  // The capture queue carries the player's hand: it runs at the highest
  // quality-of-service class, like the simulation thread. Left unspecified,
  // the system is free to schedule it as background work.
  dispatch_queue_t capture = dispatch_queue_create(
      "palmrally.camera.capture",
      dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
  AVCaptureSession* session = nil;       // touched on `control` only
  PPFrameReceiver* receiver = nil;
  CMClockRef session_clock = nullptr;    // a CoreFoundation object: retained by hand, not by ARC

  mutable std::mutex mutex;              // guards everything below
  PreviewImage preview;
  std::uint64_t preview_seq = 0;
  CameraInfo info;

  void configure_and_run();              // on `control`
  void shut_down();                      // on `control`
  void process(CMSampleBufferRef sample);  // on `capture`
  void store_preview(CVPixelBufferRef pixels);
};

void HandTrackerShared::store_preview(CVPixelBufferRef pixels) {
  CVPixelBufferLockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);
  // Two layouts arrive here. BGRA: one plane, four bytes a pixel. The
  // camera's own format: a brightness plane (one byte a pixel) followed by a
  // colour plane; the preview then uses brightness alone and is grey.
  const bool planar = CVPixelBufferIsPlanar(pixels);
  const auto* base = static_cast<const std::uint8_t*>(
      planar ? CVPixelBufferGetBaseAddressOfPlane(pixels, 0) : CVPixelBufferGetBaseAddress(pixels));
  const int w = static_cast<int>(CVPixelBufferGetWidth(pixels));
  const int h = static_cast<int>(CVPixelBufferGetHeight(pixels));
  const std::size_t stride = planar ? CVPixelBufferGetBytesPerRowOfPlane(pixels, 0)
                                    : CVPixelBufferGetBytesPerRow(pixels);

  if (base != nullptr && w > 0 && h > 0) {
    const int pw = kPreviewWidth, ph = kPreviewWidth * h / w;
    std::lock_guard lock{mutex};
    preview.width = pw;
    preview.height = ph;
    preview.rgba.resize(static_cast<std::size_t>(pw * ph) * 4);
    std::uint8_t* out = preview.rgba.data();
    for (int y = 0; y < ph; ++y) {
      const std::uint8_t* row = base + static_cast<std::size_t>(y * h / ph) * stride;
      for (int x = 0; x < pw; ++x) {
        // Nearest-neighbour shrink, mirrored left-to-right.
        const std::size_t column = static_cast<std::size_t>((pw - 1 - x) * w / pw);
        if (planar) {
          const std::uint8_t grey = row[column];
          *out++ = grey;
          *out++ = grey;
          *out++ = grey;
        } else {  // BGRA -> RGBA
          const std::uint8_t* px = row + column * 4;
          *out++ = px[2];
          *out++ = px[1];
          *out++ = px[0];
        }
        *out++ = 255;
      }
    }
    ++preview_seq;
    info.width = w;
    info.height = h;
  }
  CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);
}

void HandTrackerShared::process(CMSampleBufferRef sample) {
  if (!accepting.load(std::memory_order_acquire)) return;
  CVPixelBufferRef pixels = CMSampleBufferGetImageBuffer(sample);
  if (pixels == nullptr) return;

  HandObservation obs;
  obs.receive_time = HandTracker::now();
  // The frame's timestamp is on the capture session's clock. Convert it to
  // the host clock so it can be compared with HandTracker::now().
  CMTime pts = CMSampleBufferGetPresentationTimeStamp(sample);
  if (session_clock != nullptr) {
    pts = CMSyncConvertTime(pts, session_clock, CMClockGetHostTimeClock());
  }
  obs.capture_time = seconds(pts);
  obs.aspect = static_cast<float>(CVPixelBufferGetWidth(pixels)) /
               static_cast<float>(std::max<std::size_t>(1, CVPixelBufferGetHeight(pixels)));

  // @autoreleasepool: Vision creates temporary Objective-C objects for every
  // frame. On a dispatch queue they would otherwise pile up until the queue
  // drains its pool, which at 30+ frames a second is a visible memory saw.
  @autoreleasepool {
    PP_INTERVAL("vision hand pose");
    VNDetectHumanHandPoseRequest* request = [[VNDetectHumanHandPoseRequest alloc] init];
    request.maximumHandCount = 1;
    // Run the detector on the CPU. Left to itself Vision picks the GPU or
    // the neural engine, which is faster in isolation and slower here: the
    // game is drawing on the same GPU, and the camera's own frame delivery
    // slows down with it. Measured in this app with a hand in view:
    //   default device   delivery 55 ms + detection 16 ms = 71 ms
    //   CPU              delivery 45 ms + detection 18 ms = 63 ms
    {
      NSDictionary<VNComputeStage, NSArray<id<MLComputeDeviceProtocol>>*>* stages =
          [request supportedComputeStageDevicesAndReturnError:nil];
      for (VNComputeStage stage in stages) {
        for (id<MLComputeDeviceProtocol> device in stages[stage]) {
          if ([device isKindOfClass:[MLCPUComputeDevice class]]) {
            [request setComputeDevice:device forComputeStage:stage];
          }
        }
      }
    }
    VNImageRequestHandler* handler =
        [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixels
                                                 orientation:kCGImagePropertyOrientationUp
                                                     options:@{}];
    NSError* error = nil;
    if ([handler performRequests:@[ request ] error:&error]) {
      VNHumanHandPoseObservation* hand = request.results.firstObject;
      NSDictionary<VNHumanHandPoseObservationJointName, VNRecognizedPoint*>* points =
          hand == nil ? nil
                      : [hand recognizedPointsForJointsGroupName:
                                  VNHumanHandPoseObservationJointsGroupNameAll
                                                           error:&error];
      if (points != nil) {
        // Same order as the HandJoint enum.
        NSArray<VNHumanHandPoseObservationJointName>* names = @[
          VNHumanHandPoseObservationJointNameWrist,
          VNHumanHandPoseObservationJointNameThumbCMC, VNHumanHandPoseObservationJointNameThumbMP,
          VNHumanHandPoseObservationJointNameThumbIP, VNHumanHandPoseObservationJointNameThumbTip,
          VNHumanHandPoseObservationJointNameIndexMCP, VNHumanHandPoseObservationJointNameIndexPIP,
          VNHumanHandPoseObservationJointNameIndexDIP, VNHumanHandPoseObservationJointNameIndexTip,
          VNHumanHandPoseObservationJointNameMiddleMCP,
          VNHumanHandPoseObservationJointNameMiddlePIP,
          VNHumanHandPoseObservationJointNameMiddleDIP,
          VNHumanHandPoseObservationJointNameMiddleTip,
          VNHumanHandPoseObservationJointNameRingMCP, VNHumanHandPoseObservationJointNameRingPIP,
          VNHumanHandPoseObservationJointNameRingDIP, VNHumanHandPoseObservationJointNameRingTip,
          VNHumanHandPoseObservationJointNameLittleMCP,
          VNHumanHandPoseObservationJointNameLittlePIP,
          VNHumanHandPoseObservationJointNameLittleDIP,
          VNHumanHandPoseObservationJointNameLittleTip,
        ];
        obs.found = true;
        for (NSUInteger i = 0; i < names.count && i < kHandJointCount; ++i) {
          VNRecognizedPoint* p = points[names[i]];
          if (p == nil) continue;
          // Vision: origin bottom-left, x right. Mirror x for a mirror view.
          obs.joints[i] = {.x = 1.0f - static_cast<float>(p.location.x),
                           .y = static_cast<float>(p.location.y),
                           .confidence = p.confidence};
        }
      }
    }
  }

  // The observation goes out first; the preview copy is for the eye and can wait.
  obs.detect_time = HandTracker::now();
  if (callback && accepting.load(std::memory_order_acquire)) callback(obs);
  store_preview(pixels);
}

struct HandTracker::Impl {
  std::shared_ptr<HandTrackerShared> shared = std::make_shared<HandTrackerShared>();
};

// Picks the format with the highest frame rate at a modest resolution. For
// tracking a hand, frames per second matter far more than pixels: latency is
// bounded below by the frame interval, and detection cost grows with size.
//
// Returns with the device still LOCKED when it succeeds. On macOS a capture
// session resets the device to its own preset when it starts, unless the
// device is locked at that moment; the caller unlocks after startRunning.
static bool select_fastest_format(AVCaptureDevice* device, CameraInfo& info) {
  AVCaptureDeviceFormat* best = nil;
  AVFrameRateRange* best_range = nil;
  double best_score = -1;
  for (AVCaptureDeviceFormat* format in device.formats) {
    const CMVideoDimensions dim = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
    if (dim.width < 480 || dim.width > 1280) continue;
    for (AVFrameRateRange* range in format.videoSupportedFrameRateRanges) {
      // Frame rate first; among equals prefer the smaller image.
      const double score = range.maxFrameRate * 10000.0 - dim.width;
      if (score > best_score) {
        best_score = score;
        best = format;
        best_range = range;
      }
    }
  }
  if (best == nil) return false;

  NSError* error = nil;
  if (![device lockForConfiguration:&error]) return false;
  device.activeFormat = best;
  device.activeVideoMinFrameDuration = best_range.minFrameDuration;
  device.activeVideoMaxFrameDuration = best_range.minFrameDuration;
  info.frames_per_second = best_range.maxFrameRate;
  return true;
}

void HandTrackerShared::configure_and_run() {
  if (!wanted.load()) return;  // stop() won the race with the permission prompt

  AVCaptureDevice* device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
  if (device == nil) {
    status = CameraStatus::NoCamera;
    return;
  }
  NSError* error = nil;
  AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
  AVCaptureSession* s = [[AVCaptureSession alloc] init];
  if (input == nil || ![s canAddInput:input]) {
    status = CameraStatus::Failed;
    return;
  }

  [s beginConfiguration];
  [s addInput:input];

  AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
  // BGRA: directly readable for the preview, and Vision accepts it.
  output.videoSettings =
      @{(NSString*)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
  output.alwaysDiscardsLateVideoFrames = YES;
  receiver = [[PPFrameReceiver alloc] initWithShared:shared_from_this()];
  [output setSampleBufferDelegate:receiver queue:capture];
  if (![s canAddOutput:output]) {
    [s commitConfiguration];
    receiver = nil;
    status = CameraStatus::Failed;
    return;
  }
  [s addOutput:output];
  [s commitConfiguration];

  CameraInfo chosen;
  const bool device_locked = select_fastest_format(device, chosen);
  {
    std::lock_guard lock{mutex};
    info.frames_per_second = chosen.frames_per_second;
  }

  if (CMClockRef clock = s.synchronizationClock) {
    CFRetain(clock);
    session_clock = clock;
  }
  session = s;
  accepting.store(true, std::memory_order_release);
  [s startRunning];
  if (device_locked) [device unlockForConfiguration];
  status = s.isRunning ? CameraStatus::Running : CameraStatus::Failed;
}

void HandTrackerShared::shut_down() {
  accepting.store(false, std::memory_order_release);
  if (session != nil) {
    [session stopRunning];
    session = nil;
  }
  // Wait for a frame that is already being processed to finish, so the caller
  // can safely destroy whatever its callback touches. Only then is it safe to
  // drop the clock that frame may be using.
  dispatch_sync(capture, ^{
                });
  receiver = nil;  // also breaks the receiver <-> shared reference cycle
  if (session_clock != nullptr) {
    CFRelease(session_clock);
    session_clock = nullptr;
  }
  status = CameraStatus::Stopped;
}

HandTracker::HandTracker() : impl_(std::make_unique<Impl>()) {}

HandTracker::~HandTracker() { stop(); }

void HandTracker::start(Callback on_observation) {
  stop();
  std::shared_ptr<HandTrackerShared> shared = impl_->shared;
  shared->callback = std::move(on_observation);
  shared->wanted = true;

  const AVAuthorizationStatus auth =
      [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
  if (auth == AVAuthorizationStatusAuthorized) {
    dispatch_async(shared->control, ^{
      shared->configure_and_run();
    });
  } else if (auth == AVAuthorizationStatusNotDetermined) {
    shared->status = CameraStatus::WaitingForPermission;
    // The answer arrives whenever the user clicks, on some system thread. The
    // block holds its own shared_ptr, so it is safe even if the HandTracker
    // has been destroyed by then.
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL granted) {
                               dispatch_async(shared->control, ^{
                                 if (!shared->wanted.load()) return;
                                 if (granted) shared->configure_and_run();
                                 else shared->status = CameraStatus::Denied;
                               });
                             }];
  } else {
    shared->status = CameraStatus::Denied;
  }
}

void HandTracker::stop() {
  HandTrackerShared* shared = impl_->shared.get();
  shared->wanted = false;
  // Runs after any configure that is already queued, so start-then-stop in
  // quick succession always ends stopped.
  dispatch_sync(shared->control, ^{
    shared->shut_down();
  });
}

CameraStatus HandTracker::status() const { return impl_->shared->status.load(); }

CameraInfo HandTracker::info() const {
  std::lock_guard lock{impl_->shared->mutex};
  return impl_->shared->info;
}

bool HandTracker::copy_preview(PreviewImage& out, std::uint64_t& last_seen) const {
  std::lock_guard lock{impl_->shared->mutex};
  if (impl_->shared->preview_seq == last_seen) return false;
  out = impl_->shared->preview;
  last_seen = impl_->shared->preview_seq;
  return true;
}

double HandTracker::now() { return seconds(CMClockGetTime(CMClockGetHostTimeClock())); }

const char* describe(CameraStatus s) {
  switch (s) {
    case CameraStatus::Stopped: return "off";
    case CameraStatus::WaitingForPermission: return "waiting for permission";
    case CameraStatus::Denied: return "permission denied (System Settings > Privacy > Camera)";
    case CameraStatus::NoCamera: return "no camera found";
    case CameraStatus::Running: return "on";
    case CameraStatus::Failed: return "failed to start";
  }
  return "";
}

}  // namespace pp

@implementation PPFrameReceiver {
  std::shared_ptr<pp::HandTrackerShared> _shared;
}

- (instancetype)initWithShared:(std::shared_ptr<pp::HandTrackerShared>)shared {
  if ((self = [super init])) _shared = std::move(shared);
  return self;
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  _shared->process(sampleBuffer);
}

@end
