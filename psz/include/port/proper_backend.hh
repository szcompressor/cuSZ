#if defined(PSZ_USE_CUDA)

#define PROPER_GPU_BACKEND psz_policy::CUDA
#define PROPER_EB f8

#elif defined(PSZ_USE_HIP)

#define PROPER_GPU_BACKEND psz_policy::HIP
#define PROPER_EB f8

#elif defined(PSZ_USE_1API)

#define PROPER_GPU_BACKEND psz_policy::ONEAPI
#define PROPER_EB f4

#endif