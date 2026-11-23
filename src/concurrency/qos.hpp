#pragma once
// Thread priority on macOS.
//
// Apple silicon has fast "performance" cores and slow "efficiency" cores, and
// macOS does not let a program pin a thread to a core: there is no affinity
// API. What a thread can do is declare how urgent its work is (its quality of
// service), and the scheduler picks the cores. A thread left at the default
// may be run on an efficiency core and woken late; a "user interactive"
// thread is kept on performance cores and its timers are not coalesced.

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace pp {

// Call from the thread itself. For work whose lateness the user can feel:
// the simulation loop, benchmark threads.
inline void make_current_thread_interactive() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

inline void name_current_thread(const char* name) {
#if defined(__APPLE__)
  pthread_setname_np(name);  // shows up in Instruments and the debugger
#else
  (void)name;
#endif
}

}  // namespace pp
