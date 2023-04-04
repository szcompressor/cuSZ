#ifndef FE00973C_31E8_410D_8B42_AF65BFED4F75
#define FE00973C_31E8_410D_8B42_AF65BFED4F75

#define SETUP_ND_CPU_SERIAL                                                   \
                                                                              \
  /* fake thread-block setup */                                               \
  psz_dim3 b, t;                /* (fake) threadblock-related indices */      \
  psz_dim3 grid_dim, block_dim; /* threadblock-related dimensions */          \
                                                                              \
  /* threadblock-related strides */                                           \
  auto TWy = [&]() { return block_dim.x; };                                   \
  auto TWz = [&]() { return block_dim.x * block_dim.y; };                     \
  auto BWy = [&]() { return grid_dim.x; };                                    \
  auto BWz = [&]() { return grid_dim.x * grid_dim.y; };                       \
                                                                              \
  /* threadblock idx, linearized */                                           \
  auto bid1 = [&]() { return b.x; };                                          \
  auto bid2 = [&]() { return b.x + b.y * BWy(); };                            \
  auto bid3 = [&]() { return b.x + b.y * BWy() + b.z * BWz(); };              \
                                                                              \
  /* thread idx, linearized */                                                \
  auto tid1 = [&]() { return t.x; };                                          \
  auto tid2 = [&]() { return t.x + t.y * TWy(); };                            \
  auto tid3 = [&]() { return t.x + t.y * TWy() + t.z * TWz(); };              \
                                                                              \
  /* global data id, BLK is defined in function template */                   \
  auto gx   = [&]() { return t.x + b.x * BLK; };                              \
  auto gy   = [&]() { return t.y + b.y * BLK; };                              \
  auto gz   = [&]() { return t.z + b.z * BLK; };                              \
  auto gid1 = [&]() { return gx(); };                                         \
  auto gid2 = [&]() { return gx() + gy() * stride3.y; };                      \
  auto gid3 = [&]() { return gx() + gy() * stride3.y + gz() * stride3.z; };   \
                                                                              \
  /* partition */                                                             \
  auto data_partition = [&]() {                                               \
    grid_dim.x = (len3.x - 1) / BLK + 1, grid_dim.y = (len3.y - 1) / BLK + 1, \
    grid_dim.z  = (len3.z - 1) / BLK + 1;                                     \
    block_dim.x = BLK, block_dim.y = BLK, block_dim.z = BLK;                  \
  };                                                                          \
                                                                              \
  /* check data access validity */                                            \
  auto check_boundary1 = [&]() { return gx() < len3.x; };                     \
  auto check_boundary2 = [&]() { return gx() < len3.x and gy() < len3.y; };   \
  auto check_boundary3 = [&]() { return check_boundary2() and gz() < len3.z; };

#define PFOR1_GRID() for (b.x = 0; b.x < grid_dim.x; b.x++)
#define PFOR1_BLOCK() for (t.x = 0; t.x < BLK; t.x++)

#define PFOR2_GRID()                     \
  for (b.y = 0; b.y < grid_dim.y; b.y++) \
    for (b.x = 0; b.x < grid_dim.x; b.x++)
#define PFOR2_BLOCK()                     \
  for (t.y = 0; t.y < block_dim.y; t.y++) \
    for (t.x = 0; t.x < block_dim.x; t.x++)

#define PFOR3_GRID()                       \
  for (b.z = 0; b.z < grid_dim.z; b.z++)   \
    for (b.y = 0; b.y < grid_dim.y; b.y++) \
      for (b.x = 0; b.x < grid_dim.x; b.x++)
#define PFOR3_BLOCK()                       \
  for (t.z = 0; t.z < block_dim.z; t.z++)   \
    for (t.y = 0; t.y < block_dim.y; t.y++) \
      for (t.x = 0; t.x < block_dim.x; t.x++)

#endif /* FE00973C_31E8_410D_8B42_AF65BFED4F75 */
