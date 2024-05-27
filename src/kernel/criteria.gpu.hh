#ifndef D1A6A3E2_708C_4A61_B275_A442FBB93F19
#define D1A6A3E2_708C_4A61_B275_A442FBB93F19

namespace psz {
namespace criterion {
namespace gpu {

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

template <typename T>
struct eq {
  __device__ bool operator()(T x, T n) const { return x == n; }
};

template <typename T>
struct in_ball_shifted {
  __device__ bool operator()(T x, T radius) const
  {
    return x >= 0 and x < 2 * radius;
  }
};

template <typename T>
struct in_ball {
  __device__ bool operator()(T x, T radius) const { return fabs(x) < radius; }
};

#elif defined(PSZ_USE_1API)

template <typename T>
struct eq {
  bool operator()(T x, T n) const { return x == n; }
};

template <typename T>
struct in_ball_shifted {
  bool operator()(T x, T radius) const { return x >= 0 and x < 2 * radius; }
};

template <typename T>
struct in_ball {
  bool operator()(T x, T radius) const { return sycl::fabs(x) < radius; }
};

#endif

}  // namespace gpu
}  // namespace criterion
}  // namespace psz

#endif /* D1A6A3E2_708C_4A61_B275_A442FBB93F19 */
