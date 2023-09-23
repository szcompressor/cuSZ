#ifndef D42896C3_A5E9_427A_ABC7_C243402B6BC4
#define D42896C3_A5E9_427A_ABC7_C243402B6BC4

#define CREATE_GPUEVENT_PAIR \
  cudaEvent_t a, b;          \
  cudaEventCreate(&a);       \
  cudaEventCreate(&b);

#define DESTROY_GPUEVENT_PAIR \
  cudaEventDestroy(a);        \
  cudaEventDestroy(b);

#define START_GPUEVENT_RECORDING(STREAM) \
  cudaEventRecord(a, (cudaStream_t)STREAM);
  
#define STOP_GPUEVENT_RECORDING(STREAM)     \
  cudaEventRecord(b, (cudaStream_t)STREAM); \
  cudaEventSynchronize(b);

#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC) \
  cudaEventElapsedTime(PTR_MILLISEC, a, b);

#endif /* D42896C3_A5E9_427A_ABC7_C243402B6BC4 */
