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

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#include "kernel/detail/l23_x.cu_hip.inl"
#elif defined(PSZ_USE_1API)
#include "kernel/detail/l23_x.dp.inl"
#endif

template <typename T, typename Eq, int BLOCK, int SEQ>
bool test1d(T* data, size_t len, Eq* eq, void* stream)
{
  constexpr auto NTHREAD = BLOCK / SEQ;
  using stream_t = psz::experimental::gpu_control::stream_t;

  bool ok = true;
  auto ebx2 = 1;  // dummy

  /* init */
  for (auto i = 0; i < len; i++) data[i] = 1;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  /* process */
  psz::cuda_hip::__kernel::x_lorenzo_1d1l<T, Eq, T, BLOCK, SEQ>
      <<<1, NTHREAD, 0, (stream_t)stream>>>(
          eq, data, dim3(len, 1, 1), dim3(1, 1, 1), 0, ebx2, data);

#elif defined(PSZ_USE_1API)

  constexpr auto Tile1D = BLOCK;
  constexpr auto Seq1D = SEQ;
  auto Block1D = sycl::range<3>(1, 1, NTHREAD);
  auto Grid1D = sycl::range<3>(1, 1, 1);

  ((stream_t)stream)->submit([&](sycl::handler& cgh) {
    constexpr auto NTHREAD = Tile1D / Seq1D;
    sycl::local_accessor<T, 1> scratch(sycl::range<1>(Tile1D), cgh);
    sycl::local_accessor<Eq, 1> s_eq(sycl::range<1>(Tile1D), cgh);
    sycl::local_accessor<T, 1> exch_in(sycl::range<1>(NTHREAD / 32), cgh);
    sycl::local_accessor<T, 1> exch_out(sycl::range<1>(NTHREAD / 32), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(Grid1D * Block1D, Block1D),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          psz::dpcpp::__kernel::x_lorenzo_1d1l<T, Eq, T, Tile1D, Seq1D>(
              eq, data, sycl::range<3>(1, 1, len), sycl::range<3>(1, 1, 1), 0,
              ebx2, data, item_ct1, scratch.get_pointer(), s_eq.get_pointer(),
              exch_in.get_pointer(), exch_out.get_pointer());
        });
  });

#endif
  psz::experimental::stream_sync(stream);

  /* verify */
  for (auto i = 0; i < len; i++) {
    if (data[i] != i + 1 /* inclusive scan */) {
      ok = false;
      goto END;
    }
  }

END:
  return ok;
}

template <typename T, typename Eq>
bool test2d(T* data, size_t len, Eq* eq, void* stream)
{
  using stream_t = psz::experimental::gpu_control::stream_t;
  constexpr auto BLOCK = 16;

  bool ok = true;
  auto ebx2 = 1;  // dummy

  /* init */
  for (auto i = 0; i < len; i++) data[i] = 1;

    /* process */
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  psz::cuda_hip::__kernel::x_lorenzo_2d1l<T, Eq, T>
      <<<dim3(1, 1, 1), dim3(16, 2, 1), 0, (stream_t)stream>>>(
          eq, data, dim3(16, 16, 1), dim3(1, 16, 1), 0, ebx2, data);
#elif defined(PSZ_USE_1API)

  auto Block2D = sycl::range<3>(1, 2, 16);
  auto Grid2D = sycl::range<3>(1, 1, 1);

  ((stream_t)stream)->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> scratch(sycl::range<1>(16 /*BLOCK*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(Grid2D * Block2D, Block2D),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          psz::dpcpp::__kernel::x_lorenzo_2d1l<T, Eq, T>(
              eq, data, sycl::range<3>(1, 16, 16), sycl::range<3>(1, 16, 1), 0,
              ebx2, data, item_ct1, scratch.get_pointer());
        });
  });

#endif

  psz::experimental::stream_sync(stream);

  /* verify */
  for (auto y = 0; y < BLOCK; y++) {
    for (auto x = 0; y < BLOCK; y++) {
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

template <typename T, typename Eq>
bool test3d(T* data, size_t len, Eq* eq, void* stream)
{
  using stream_t = psz::experimental::gpu_control::stream_t;
  constexpr auto BLOCK = 8;

  bool ok = true;
  auto ebx2 = 1;  // dummy

  /* init */
  for (auto i = 0; i < len; i++) data[i] = 1;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  /* process */
  psz::cuda_hip::__kernel::x_lorenzo_3d1l<T, Eq, T>
      <<<dim3(1, 1, 1), dim3(32, 1, 8), 0, (stream_t)stream>>>(
          eq, data, dim3(32, 8, 8), dim3(1, 32, 32 * 8), 0, ebx2, data);
#elif defined(PSZ_USE_1API)

  auto Block3D = sycl::range<3>(8, 1, 32);
  auto Grid3D = sycl::range<3>(1, 1, 1);

  ((stream_t)stream)->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 3> scratch(sycl::range<3>(8 /*BLOCK*/, 4, 8), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(Grid3D * Block3D, Block3D),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          psz::dpcpp::__kernel::x_lorenzo_3d1l<T, Eq, T>(
              eq, data, sycl::range<3>(8, 8, 32),
              sycl::range<3>(32 * 8, 32, 1), 0, ebx2, data, item_ct1, scratch);
        });
  });

#endif
  psz::experimental::stream_sync(stream);

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

template <int dimension, int BLOCK_1D = 256, int SEQ_1D = 8>
bool test_inclscan_1block()
{
  using T = float;
  using Eq = uint32_t;
  using FP = T;

  auto len = 1;
  if (dimension == 1)
    len = BLOCK_1D;
  else if (dimension == 2)
    len = 16 * 16;
  else if (dimension == 3)
    len = 32 * 8 * 8;

  PSZ_DEFAULT_CREATE_STREAM(stream);

  auto data = psz::experimental::malloc_shared<T>(len, stream);
  auto eq = psz::experimental::malloc_shared<Eq>(len, stream);

  bool ok = true;

  printf("test len: %u\t", len);

  if (dimension == 1)
    ok = ok && test1d<T, Eq, BLOCK_1D, SEQ_1D>(data, len, eq, stream);
  else if (dimension == 2)
    ok = ok && test2d<T, Eq>(data, len, eq, stream);
  else if (dimension == 3)
    ok = ok && test3d<T, Eq>(data, len, eq, stream);

  psz::experimental::free_shared(data, stream);
  psz::experimental::free_shared(eq, stream);

  PSZ_DEFAULT_DETELE_STREAM(stream);

  return ok;
}