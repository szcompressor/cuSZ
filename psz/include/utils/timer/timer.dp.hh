#ifndef AC21DA73_7451_4FC7_8064_0B3206128605
#define AC21DA73_7451_4FC7_8064_0B3206128605

#define SYCL_TIME_DELTA(EVENT, MILLISEC)                                    \
  auto sycl_time_delta = [](sycl::event& e) {                               \
    cl_ulong start_time =                                                   \
        e.get_profiling_info<sycl::info::event_profiling::command_start>(); \
    cl_ulong end_time =                                                     \
        e.get_profiling_info<sycl::info::event_profiling::command_end>();   \
    return ((float)(end_time - start_time)) / 1e6;                          \
  };                                                                        \
  MILLISEC = sycl_time_delta(EVENT);

#define CREATE_GPUEVENT_PAIR      \
  auto start = new sycl::event(); \
  auto end = new sycl::event();

#define DESTROY_GPUEVENT_PAIR \
  dpct::destroy_event(start); \
  dpct::destroy_event(end);

#define START_GPUEVENT_RECORDING(STREAM) \
  *start = ((sycl::queue*)STREAM)->ext_oneapi_submit_barrier();

#define STOP_GPUEVENT_RECORDING(STREAM)                       \
  *end = ((sycl::queue*)STREAM)->ext_oneapi_submit_barrier(); \
  end->wait_and_throw();

#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC)                                  \
  *PTR_MILLISEC =                                                            \
      (end->get_profiling_info<sycl::info::event_profiling::command_end>() - \
       start->get_profiling_info<                                            \
           sycl::info::event_profiling::command_start>()) /                  \
      1e3f;

#endif /* AC21DA73_7451_4FC7_8064_0B3206128605 */
