#ifndef C0AB2113_D15A_4982_A413_2C37DB993EB7
#define C0AB2113_D15A_4982_A413_2C37DB993EB7

#define CREATE_GPUEVENT_PAIR \
  hipEvent_t a, b;           \
  hipEventCreate(&a);        \
  hipEventCreate(&b);

#define DESTROY_GPUEVENT_PAIR \
  hipEventDestroy(a);         \
  hipEventDestroy(b);

#define START_GPUEVENT_RECORDING(STREAM) \
  hipEventRecord(a, (hipStream_t)STREAM);

#define STOP_GPUEVENT_RECORDING(STREAM)   \
  hipEventRecord(b, (hipStream_t)STREAM); \
  hipEventSynchronize(b);

#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC) \
  hipEventElapsedTime(PTR_MILLISEC, a, b);

#endif /* C0AB2113_D15A_4982_A413_2C37DB993EB7 */
