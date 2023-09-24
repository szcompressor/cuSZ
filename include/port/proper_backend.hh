#if defined(PSZ_USE_CUDA)
#define PROPER_GPU_BACKEND pszpolicy::CUDA
#elif defined(PSZ_USE_HIP)
#define PROPER_GPU_BACKEND pszpolicy::HIP
#elif defined(PSZ_USE_1API)
#define PROPER_GPU_BACKEND pszpolicy::ONEAPI
#endif