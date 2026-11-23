cmake_minimum_required(VERSION 3.28)
project(palmrally VERSION 0.1.0 LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Older third-party CMakeLists declare minimums that CMake 4 no longer accepts.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ---------------------------------------------------------------- dependencies
include(FetchContent)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG 5.5
  GIT_SHALLOW TRUE
  SYSTEM)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1
  GIT_SHALLOW TRUE
  SYSTEM)
FetchContent_MakeAvailable(raylib Catch2)

# Warnings apply to our code only, never to dependencies.
add_library(pp_warnings INTERFACE)
target_compile_options(pp_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wdouble-promotion -Werror)

# ------------------------------------------------------------------------ core
# Pure C++: no Apple APIs, no raylib. Everything here is testable headless.
add_library(pp_core STATIC
  src/physics/ball.cpp
  src/physics/world.cpp
  src/game/shot.cpp
  src/game/match.cpp
  src/game/sim_thread.cpp
  src/game/player_paddle.cpp
  src/game/opponent_planner.cpp
  src/game/telemetry.cpp
  src/game/session_log.cpp
  src/planner/shot_batch.cpp
  src/estimation/palm_filter.cpp)
target_include_directories(pp_core PUBLIC src)
target_link_libraries(pp_core PRIVATE pp_warnings)

# -------------------------------------------------------------------- platform
# The only target that touches Apple frameworks. One Objective-C++ file behind
# a plain C++ header; -fobjc-arc turns on automatic reference counting for the
# Objective-C objects in it.
add_library(pp_platform STATIC src/platform/hand_tracker.mm)
target_include_directories(pp_platform PUBLIC src)
target_compile_options(pp_platform PRIVATE -fobjc-arc)
target_link_libraries(pp_platform
  PUBLIC "-framework AVFoundation" "-framework Vision" "-framework CoreML" "-framework CoreMedia"
         "-framework CoreVideo" "-framework Foundation"
  PRIVATE pp_warnings)

# ---------------------------------------------------------------------- render
# The Metal renderer: Objective-C++ behind a plain C++ header, like the camera.
add_library(pp_render STATIC src/render/metal_renderer.mm)
target_include_directories(pp_render PUBLIC src)
target_compile_options(pp_render PRIVATE -fobjc-arc)
target_link_libraries(pp_render
  PUBLIC "-framework Metal" "-framework MetalKit" "-framework QuartzCore" "-framework CoreText"
         "-framework CoreGraphics" "-framework ImageIO" "-framework Cocoa"
  PRIVATE pp_warnings)

# Sound. raylib is used for its audio mixer only, and only inside this file.
add_library(pp_audio STATIC src/audio/sounds.cpp)
target_include_directories(pp_audio PUBLIC src)
target_link_libraries(pp_audio PRIVATE raylib pp_warnings)

# ------------------------------------------------------------------------- app
# The game: a Cocoa window drawn with Metal.
add_executable(palmrally MACOSX_BUNDLE app/metal_main.mm app/game_app.cpp)
target_include_directories(palmrally PRIVATE app)
target_compile_options(palmrally PRIVATE -fobjc-arc)
target_link_libraries(palmrally PRIVATE pp_core pp_platform pp_render pp_audio pp_warnings)
set_target_properties(palmrally PROPERTIES
  OUTPUT_NAME "Palm Rally"
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/app/Info.plist.in
  MACOSX_BUNDLE_BUNDLE_NAME "Palm Rally"
  MACOSX_BUNDLE_GUI_IDENTIFIER "local.palmrally"
  MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION}
  MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION})

# The first renderer (raylib, OpenGL), kept to compare frame pacing against.
add_executable(palmrally_gl app/main.cpp)
target_link_libraries(palmrally_gl PRIVATE pp_core pp_platform raylib pp_warnings)

# ----------------------------------------------------------------------- tools
add_executable(feature_probe tools/feature_probe.cpp)

# Camera pipeline check. A bare executable has no app bundle, so the usage
# description macOS requires for camera access is embedded in the binary.
add_executable(hand_probe tools/hand_probe.cpp)
target_link_libraries(hand_probe PRIVATE pp_core pp_platform pp_warnings)
target_link_options(hand_probe PRIVATE
  "LINKER:-sectcreate,__TEXT,__info_plist,${CMAKE_SOURCE_DIR}/tools/probe_Info.plist")

# Summarises a recorded play session: the numbers used to tune the game.
add_executable(session_report tools/session_report.cpp)
target_link_libraries(session_report PRIVATE pp_core pp_warnings)

# Replays a session's camera frames through the current hand control and
# measures the result: delay, shake, how much the game moves the bat.
add_executable(session_replay tools/session_replay.cpp)
target_link_libraries(session_replay PRIVATE pp_core pp_warnings)

# ------------------------------------------------------------------ benchmarks
add_executable(queue_bench bench/queue_bench.cpp)
target_link_libraries(queue_bench PRIVATE pp_core pp_warnings)
add_executable(planner_bench bench/planner_bench.cpp)
target_link_libraries(planner_bench PRIVATE pp_core pp_warnings)
add_executable(sim_bench bench/sim_bench.cpp)
target_link_libraries(sim_bench PRIVATE pp_core pp_warnings)

# ----------------------------------------------------------------------- tests
enable_testing()
add_executable(pp_tests
  tests/math/vec3_test.cpp
  tests/physics/ball_test.cpp
  tests/physics/world_test.cpp
  tests/game/shot_test.cpp
  tests/game/match_test.cpp
  tests/estimation/palm_test.cpp
  tests/concurrency/concurrency_test.cpp
  tests/estimation/palm_filter_test.cpp
  tests/game/player_paddle_test.cpp
  tests/game/telemetry_test.cpp
  tests/planner/shot_batch_test.cpp
  tests/game/swing_test.cpp
  tests/game/bat_anchor_test.cpp
  tests/game/bat_smoother_test.cpp
  tests/game/bat_face_test.cpp
  tests/game/hand_control_test.cpp
  tests/render/mat4_test.cpp)
target_link_libraries(pp_tests PRIVATE pp_core Catch2::Catch2WithMain pp_warnings)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(pp_tests)

# Its own program: it replaces the global operator new to count allocations.
add_executable(no_alloc_test tests/alloc/no_alloc_test.cpp)
target_link_libraries(no_alloc_test PRIVATE pp_core pp_warnings)
add_test(NAME "the simulation loop never allocates" COMMAND no_alloc_test)
