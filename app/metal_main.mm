// Palm Rally: the window, its events, and the display-rate callback.
//
// Three threads:
//   capture thread   camera frames -> hand pose         (src/platform)
//   sim thread       physics and rules at 1 kHz         (src/game/sim_thread)
//   main thread      this file + app/game_app.cpp: input, drawing, sound
// The main thread never touches the simulation. It sends small commands in
// and draws whatever snapshot the simulation published last.
//
// This file is the only part of the app written against Cocoa. MTKView calls
// drawInMTKView: once per display refresh (it is driven by the display's own
// timer, so frames are paced to the screen, 120 Hz on a ProMotion panel).
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

#include "game_app.hpp"
#include "render/metal_renderer.hpp"

#include <cstdlib>
#include <memory>
#include <string_view>

@interface GameView : MTKView
@property(nonatomic) pp::GameApp* game;
@end

@implementation GameView
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (void)keyDown:(NSEvent*)event {
  NSString* chars = event.charactersIgnoringModifiers.lowercaseString;
  if (self.game != nullptr && chars.length > 0 && !event.isARepeat) {
    const unichar c = [chars characterAtIndex:0];
    if (c < 128) self.game->key_pressed(static_cast<char>(c));
  }
}
- (void)track:(NSEvent*)event {
  if (self.game == nullptr) return;
  const NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
  // Cocoa counts y upward from the bottom of the view.
  self.game->mouse_moved(static_cast<float>(p.x), static_cast<float>(self.bounds.size.height - p.y));
}
- (void)mouseMoved:(NSEvent*)event {
  [self track:event];
}
- (void)mouseDragged:(NSEvent*)event {
  [self track:event];
}
@end

@interface AppDelegate : NSObject <NSApplicationDelegate, MTKViewDelegate>
- (instancetype)initWithOptions:(pp::AppOptions)options;
@end

@implementation AppDelegate {
  pp::AppOptions _options;
  NSWindow* _window;
  GameView* _view;
  std::unique_ptr<pp::render::MetalRenderer> _renderer;
  std::unique_ptr<pp::GameApp> _game;
  CFTimeInterval _last;
}

- (instancetype)initWithOptions:(pp::AppOptions)options {
  if ((self = [super init])) _options = options;
  return self;
}

- (void)applicationDidFinishLaunching:(NSNotification*)note {
  const NSRect frame = NSMakeRect(0, 0, 1280, 720);
  _window = [[NSWindow alloc]
      initWithContentRect:frame
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                          NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                  backing:NSBackingStoreBuffered
                    defer:NO];
  _window.title = @"Palm Rally";
  _window.acceptsMouseMovedEvents = YES;
  _window.restorable = NO;
  [_window center];

  _view = [[GameView alloc] initWithFrame:frame device:MTLCreateSystemDefaultDevice()];
  _renderer = std::make_unique<pp::render::MetalRenderer>((__bridge void*)_view);
  _game = std::make_unique<pp::GameApp>(*_renderer, _options);
  _view.game = _game.get();
  _view.delegate = self;
  _view.preferredFramesPerSecond = 120;
  _window.contentView = _view;
  [_window makeKeyAndOrderFront:nil];
  [_window makeFirstResponder:_view];
  [NSApp activateIgnoringOtherApps:YES];
  _last = CACurrentMediaTime();
}

- (void)drawInMTKView:(MTKView*)view {
  const CFTimeInterval now = CACurrentMediaTime();
  _game->frame(static_cast<float>(now - _last));
  _last = now;
  if (_game->wants_quit()) [NSApp terminate:nil];
}

- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
  return YES;
}

- (void)applicationWillTerminate:(NSNotification*)note {
  // Stop drawing, then tear down in order: the game (camera, simulation,
  // session file) before the renderer it draws with.
  _view.paused = YES;
  _view.delegate = nil;
  _view.game = nullptr;
  _game.reset();
  _renderer.reset();
}
@end

int main(int argc, char** argv) {
  // --screenshot <file.png>: render for a while, save one frame, exit.
  // --camera: start with the camera on.   --no-record: do not write a session log.
  pp::AppOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--screenshot" && i + 1 < argc) options.screenshot = argv[i + 1];
    if (arg == "--screenshot-frame" && i + 1 < argc) options.screenshot_frame = std::atoi(argv[i + 1]);
    if (arg == "--bench" && i + 1 < argc) {
      options.bench_frames = std::atoi(argv[i + 1]);
      options.demo = true;
      options.record = false;
    }
    if (arg == "--demo") options.demo = true;
    if (arg == "--camera") options.camera = true;
    if (arg == "--no-record") options.record = false;
  }

  @autoreleasepool {
    // No "reopen the windows from last time?": there is one window and
    // nothing in it to restore. (Without this, macOS asks, after any run that
    // did not quit cleanly, with a dialog that blocks the first frame.)
    [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"ApplePersistenceIgnoreState"];
    [[NSUserDefaults standardUserDefaults] setBool:NO forKey:@"NSQuitAlwaysKeepsWindows"];
    NSApplication* app = [NSApplication sharedApplication];
    app.activationPolicy = NSApplicationActivationPolicyRegular;

    // A minimal menu, so Cmd-Q works.
    NSMenu* bar = [NSMenu new];
    NSMenuItem* item = [NSMenuItem new];
    [bar addItem:item];
    NSMenu* menu = [NSMenu new];
    [menu addItemWithTitle:@"Quit Palm Rally" action:@selector(terminate:) keyEquivalent:@"q"];
    item.submenu = menu;
    app.mainMenu = bar;

    AppDelegate* delegate = [[AppDelegate alloc] initWithOptions:options];
    app.delegate = delegate;
    [app run];
  }
  return 0;
}
