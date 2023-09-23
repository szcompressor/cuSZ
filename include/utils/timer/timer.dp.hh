#ifndef AC21DA73_7451_4FC7_8064_0B3206128605
#define AC21DA73_7451_4FC7_8064_0B3206128605

#define CREATE_GPUEVENT_PAIR  \
  dpct::event_ptr start, end; \
  start = new sycl::event();  \
  end = new sycl::event();

#define DESTROY_GPUEVENT_PAIR \
  dpct::destroy_event(start); \
  dpct::destroy_event(end);

#define START_GPUEVENT_RECORDING(STREAM) \
  *start = ((dpct::queue_ptr)STREAM)->ext_oneapi_submit_barrier();

#define STOP_GPUEVENT_RECORDING(STREAM)                          \
  *end = ((dpct::queue_ptr)STREAM)->ext_oneapi_submit_barrier(); \
  end->wait_and_throw();

#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC)                                  \
  *PTR_MILLISEC =                                                            \
      (end->get_profiling_info<sycl::info::event_profiling::command_end>() - \
       start->get_profiling_info<                                            \
           sycl::info::event_profiling::command_start>()) /                  \
      1e3f;

#endif /* AC21DA73_7451_4FC7_8064_0B3206128605 */
