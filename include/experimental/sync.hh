namespace psz {
namespace experimental {

using queue_ptr = void*;

template <psz_platform P = PROPER_GPU_BACKEND>
void device_sync(queue_ptr stream = nullptr)
{
#if defined(PSZ_USE_CUDA)
  cudaDeviceSynchronize();
#elif defined(PSZ_USE_HIP)
  hipDeviceSynchronize();
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend "
        "does not allow stream to be null.");
  ((sycl::queue*)stream)->wait_and_throw();
#endif
}

template <psz_platform P = PROPER_GPU_BACKEND>
void stream_sync(queue_ptr stream)
{
#if defined(PSZ_USE_CUDA)
  cudaStreamSynchronize((cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
  hipStreamSynchronize((hipStream_t)stream);
#elif defined(PSZ_USE_1API)
  ((sycl::queue*)stream)->wait_and_throw();
#endif
}

}  // namespace experimental
}  // namespace psz
