// Reports which C++23 library features this toolchain actually ships, and the
// real cache-line size of this machine. Run once; results steer M3.
#include <cstddef>
#include <cstdio>
#include <new>
#include <version>

#include <sys/sysctl.h>

#define REPORT(macro_name, value) std::printf("  %-48s %ld\n", macro_name, static_cast<long>(value))

int main() {
  std::printf("__cplusplus = %ld\n\nLibrary feature macros (0 = missing):\n", static_cast<long>(__cplusplus));

#ifdef __cpp_lib_jthread
  REPORT("__cpp_lib_jthread", __cpp_lib_jthread);
#else
  REPORT("__cpp_lib_jthread", 0);
#endif
#ifdef __cpp_lib_expected
  REPORT("__cpp_lib_expected", __cpp_lib_expected);
#else
  REPORT("__cpp_lib_expected", 0);
#endif
#ifdef __cpp_lib_print
  REPORT("__cpp_lib_print", __cpp_lib_print);
#else
  REPORT("__cpp_lib_print", 0);
#endif
#ifdef __cpp_lib_mdspan
  REPORT("__cpp_lib_mdspan", __cpp_lib_mdspan);
#else
  REPORT("__cpp_lib_mdspan", 0);
#endif
#ifdef __cpp_lib_atomic_wait
  REPORT("__cpp_lib_atomic_wait", __cpp_lib_atomic_wait);
#else
  REPORT("__cpp_lib_atomic_wait", 0);
#endif
#ifdef __cpp_lib_memory_resource
  REPORT("__cpp_lib_memory_resource", __cpp_lib_memory_resource);
#else
  REPORT("__cpp_lib_memory_resource", 0);
#endif
#ifdef __cpp_lib_hardware_interference_size
  REPORT("__cpp_lib_hardware_interference_size", __cpp_lib_hardware_interference_size);
  REPORT("  hardware_destructive_interference_size", std::hardware_destructive_interference_size);
  REPORT("  hardware_constructive_interference_size", std::hardware_constructive_interference_size);
#else
  REPORT("__cpp_lib_hardware_interference_size", 0);
#endif

  long line = 0;
  std::size_t len = sizeof(line);
  if (sysctlbyname("hw.cachelinesize", &line, &len, nullptr, 0) == 0) {
    std::printf("\nsysctl hw.cachelinesize = %ld bytes\n", line);
  }
  return 0;
}
