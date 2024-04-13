#ifndef A3C56F10_4896_4902_AD5B_6D11E4D09D64
#define A3C56F10_4896_4902_AD5B_6D11E4D09D64

struct gpu_control {
#if defined(PSZ_USE_CUDA)
  using stream_t = cudaStream_t;
#elif defined(PSZ_USE_HIP)
  using stream_t = hipStream_t;
#elif defined(PSZ_USE_1API)
  using stream_t = sycl::queue*;
#endif
};

#endif /* A3C56F10_4896_4902_AD5B_6D11E4D09D64 */
