
#if defined(PSZ_USE_CUDA)

#define PSZ_DEFAULT_CREATE_STREAM(STREAM_VAR) \
  cudaStream_t STREAM_VAR;                    \
  cudaStreamCreate(&STREAM_VAR);

#define PSZ_DEFAULT_DETELE_STREAM(STREAM_VAR) cudaStreamDestroy(STREAM_VAR);

#elif defined(PSZ_USE_HIP)

#define PSZ_DEFAULT_CREATE_STREAM(STREAM_VAR) \
  hipStream_t stream;                         \
  hipStreamCreate(&stream);

#define PSZ_DEFAULT_DETELE_STREAM(STREAM_VAR) hipStreamDestroy(STREAM_VAR);

#elif defined(PSZ_USE_1API)

#define PSZ_DEFAULT_CREATE_STREAM(STREAM_VAR)     \
  auto plist = sycl::property_list(               \
      sycl::property::queue::in_order(),          \
      sycl::property::queue::enable_profiling()); \
  auto STREAM_VAR = new sycl::queue(sycl::default_selector_v, plist);

#define PSZ_DEFAULT_DETELE_STREAM(STREAM_VAR) delete STREAM_VAR;

#endif
