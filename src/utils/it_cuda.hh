#ifndef D5AFF1E5_1D3A_4FB8_A8DF_3785847FCBF4
#define D5AFF1E5_1D3A_4FB8_A8DF_3785847FCBF4

#define SETUP_ND_GPU_CUDA                                                   \
                                                                            \
  /* threadblock-related indices */                                         \
  auto t = [&]() -> dim3 { return threadIdx; };                             \
  auto b = [&]() -> dim3 { return blockIdx; };                              \
  /* threadblock-related indices, freely mapping 1D to ND */                \
  auto tx = [&]() { return threadIdx.x; };                                  \
  auto ty = [&](auto stridey) { return threadIdx.x / stridey; };            \
  auto tz = [&](auto stridez) { return threadIdx.x / stridez; };            \
  /* threadblock-related dimensions */                                      \
  auto BD = [&]() -> dim3 { return blockDim; };                             \
  auto GD = [&]() -> dim3 { return gridDim; };                              \
  /* threadblock-related strides */                                         \
  auto TWy = [&]() { return blockDim.x; };                                  \
  auto TWz = [&]() { return blockDim.x * blockDim.y; };                     \
  auto BWy = [&]() { return gridDim.x; };                                   \
  auto BWz = [&]() { return gridDim.x * gridDim.y; };                       \
                                                                            \
  /* threadblock idx, linearized */                                         \
  auto bid1 = [&]() { return b().x; };                                      \
  auto bid2 = [&]() { return b().x + b().y * BWy(); };                      \
  auto bid3 = [&]() { return b().x + b().y * BWy() + b().z * BWz(); };      \
                                                                            \
  /* thread idx, linearized */                                              \
  auto tid1 = [&]() { return t().x; };                                      \
  auto tid2 = [&]() { return t().x + t().y * TWy(); };                      \
  auto tid3 = [&]() { return t().x + t().y * TWy() + t().z * TWz(); };      \
                                                                            \
  /* global data id, BLK is defined in function template */                 \
  auto gx   = [&]() { return t().x + b().x * BLK; };                        \
  auto gy   = [&]() { return t().y + b().y * BLK; };                        \
  auto gz   = [&]() { return t().z + b().z * BLK; };                        \
  auto gid1 = [&]() { return gx(); };                                       \
  auto gid2 = [&]() { return gx() + gy() * stride3.y; };                    \
  auto gid3 = [&]() { return gx() + gy() * stride3.y + gz() * stride3.z; }; \
                                                                            \
  /* check data access validity */                                          \
  auto check_boundary1 = [&]() { return gx() < len3.x; };                   \
  auto check_boundary2 = [&]() { return gx() < len3.x and gy() < len3.y; }; \
  auto check_boundary3 = [&]() { return check_boundary2() and gz() < len3.z; };

#endif /* D5AFF1E5_1D3A_4FB8_A8DF_3785847FCBF4 */
