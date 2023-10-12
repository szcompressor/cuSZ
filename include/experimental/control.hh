

namespace psz::experimental {

struct gpu_control {
#if defined(PSZ_USE_CUDA)
  using stream_t = cudaStream_t;
#elif defined(PSZ_USE_HIP)
  using stream_t = hipStream_t;
#elif defined(PSZ_USE_1API)
  using stream_t = sycl::queue*;
#endif
};

}  // namespace psz::experimental
