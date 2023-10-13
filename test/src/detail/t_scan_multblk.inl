/**
 * @file test_l1_l23scan.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-23
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "experimental/p9y.hh"
#include "kernel/lrz/l23.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#include "kernel/detail/l23_x.cu_hip.inl"
#elif defined(PSZ_USE_1API)
#include "kernel/detail/l23_x.dp.inl"
#endif

template <typename T, typename Eq = u4>
bool test1d_multblk_verify(T* data, size_t len, Eq* eq)
{
  bool ok = true;

  /* verify */
  for (auto i = 0; i < len; i++) {
    if (data[i] != (i) + 1 /* inclusive scan */) {
      ok = false;
      goto END;
    }
  }

END:
  return ok;
}

template <typename T, typename Eq = u4>
bool test2d_multblk_verify(T* data, size_t len, Eq* eq)
{
  bool ok = true;

  /* verify */
  for (auto y = 0; y < 16; y++) {
    for (auto x = 0; y < 16; y++) {
      auto id = x + y * 16;
      auto supposed = (x + 1) * (y + 1);
      if (data[id] != supposed) {
        ok = false;
        goto END;
      }
    }
  }
END:
  return ok;
}

template <typename T, typename Eq = u4>
bool test3d_multblk_verify(T* data, size_t len, Eq* eq)
{
  using stream_t = psz::experimental::gpu_control::stream_t;
  constexpr auto BLOCK = 8;

  bool ok = true;
  auto ebx2 = 1;  // dummy

  /* init */
  for (auto i = 0; i < len; i++) data[i] = 1;

  /* verify */
  for (auto z = 0; z < BLOCK; z++) {
    for (auto y = 0; y < BLOCK; y++) {
      for (auto real_x = 0; real_x < 4 * BLOCK; real_x++) {
        auto x = (real_x % BLOCK);
        auto id = real_x + (y * 32) + (z * 32 * 8);
        auto supposed = (x + 1) * (y + 1) * (z + 1);

        if (data[id] != supposed) {
          ok = false;
          goto END;
        }
      }
    }
  }

END:
  return ok;
}

bool test_inclscan_multipleblock(size_t x, size_t y, size_t z)
{
  using T = float;
  using Eq = uint32_t;
  using FP = T;
  using core_type = psz::experimental::core_type;
  using stream_t = psz::experimental::gpu_control::stream_t;

  float time;

  PSZ_DEFAULT_CREATE_STREAM(stream);

  auto len3 = core_type::range3(x, y, z);
  auto len = core_type::linearize(len3);

  auto data = psz::experimental::malloc_shared<T>(len, stream);
  auto eq = psz::experimental::malloc_shared<Eq>(len, stream);

  for (auto i = 0; i < len; i++) data[i] = 1;

  bool ok = true;

  printf("test len: %u\t", 1);
  psz_decomp_l23(eq, len3, data, 0.5, 0, data, &time, stream);

  psz::experimental::free_shared(data, stream);
  psz::experimental::free_shared(eq, stream);

  PSZ_DEFAULT_DETELE_STREAM(stream);

  return ok;
}