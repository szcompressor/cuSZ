/**
 * @file interp_spline3.cuh
 * @author Jiannan Tian
 * @brief (header) A high-level Spline3D wrapper. Allocations are explicitly
 * out of called functions.
 * @version 0.3
 * @date 2021-06-15
 * (rev) 2023-08-04
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_INTERP_SPLINE_CUH
#define CUSZ_WRAPPER_INTERP_SPLINE_CUH

#include <exception>
#include <iostream>
#include <limits>
#include <numeric>

#include "kernel/detail/spline3.inl"
#include "mem/memseg_cxx.hh"

using std::cout;

namespace cusz {

template <typename T, typename E, typename FP = T>
class Spline3 {
 public:
  using Precision = FP;

 private:
  static const auto BLOCK = 8;

  using TITER = T*;
  using EITER = E*;

 public:
  bool dbg_mode{false};

  uint32_t dimx, dimx_aligned, nblockx, nanchorx;
  uint32_t dimy, dimy_aligned, nblocky, nanchory;
  uint32_t dimz, dimz_aligned, nblockz, nanchorz;
  uint32_t len, len_aligned, len_anchor;
  dim3 data_size, size_aligned, data_leap, leap_aligned, anchor_size,
      anchor_leap;

  float time_elapsed;

  float get_time_elapsed() const { return time_elapsed; }
  uint32_t get_workspace_nbyte() const { return 0; };

  pszmem_cxx<T>* anchor;
  pszmem_cxx<E>* ectrl;
  pszmem_cxx<T>* outlier;

 public:
  Spline3() = default;

  Spline3(dim3 _size, bool _dbg_mode = false) :
      data_size(_size), dbg_mode(_dbg_mode)
  {
    auto debug = [&]() {
      printf("\ndebug in spline3::constructor\n");
      printf("dim.xyz & len:\t%d, %d, %d, %d\n", dimx, dimy, dimz, len);
      printf("nblock.xyz:\t%d, %d, %d\n", nblockx, nblocky, nblockz);
      printf(
          "aligned.xyz:\t%d, %d, %d\n", dimx_aligned, dimy_aligned,
          dimz_aligned);
      printf("nanchor.xyz:\t%d, %d, %d\n", nanchorx, nanchory, nanchorz);
      printf("data_len:\t%d\n", len);
      printf("anchor_len:\t%d\n", len_anchor);
      printf("quant_len:\t%d\n", len_aligned);
      printf("quant_footprint:\t%d\n", len_aligned);
      printf("NBYTE anchor:\t%lu\n", sizeof(T) * len_anchor);
      printf("NBYTE ectrl:\t%lu\n", sizeof(E) * len_aligned);
      cout << '\n';
    };

    // original data_size
    dimx = data_size.x, dimy = data_size.y, dimz = data_size.z;
    len = dimx * dimy * dimz;

    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

    // partition & aligning
    nblockx = div(dimx, BLOCK * 4);
    nblocky = div(dimy, BLOCK);
    nblockz = div(dimz, BLOCK);
    dimx_aligned = nblockx * 32;  // 235 -> 256
    dimy_aligned = nblocky * 8;   // 449 -> 456
    dimz_aligned = nblockz * 8;   // 449 -> 456
    len_aligned = dimx_aligned * dimy_aligned * dimz_aligned;

    // multidimensional
    data_leap = dim3(1, dimx, dimx * dimy);
    size_aligned = dim3(dimx_aligned, dimy_aligned, dimz_aligned);
    leap_aligned = dim3(1, dimx_aligned, dimx_aligned * dimy_aligned);

    // anchor point
    nanchorx = int(dimx / BLOCK) + 1;
    nanchory = int(dimy / BLOCK) + 1;
    nanchorz = int(dimz / BLOCK) + 1;
    len_anchor = nanchorx * nanchory * nanchorz;
    anchor_size = dim3(nanchorx, nanchory, nanchorz);
    anchor_leap = dim3(1, nanchorx, nanchorx * nanchory);

    if (dbg_mode) debug();
  }

  void allocate()
  {
    // allocate
    anchor = new pszmem_cxx<T>(len_anchor, 1, 1, "anchor");
    ectrl = new pszmem_cxx<E>(len_aligned, 1, 1, "ectrl");
    anchor->control({Malloc});
    ectrl->control({Malloc});
  }

  ~Spline3()
  {
    delete anchor;
    delete ectrl;
  }

  static int construct(
      TITER d_data, dim3 data_size, dim3 data_leap, double const eb,
      int const radius, EITER d_ectrl, dim3 size_aligned, dim3 leap_aligned,
      TITER d_anchor, dim3 anchor_leap, cudaStream_t stream)
  {
    constexpr auto BLOCK = 8;
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    auto nblockx = div(data_size.x, BLOCK * 4);
    auto nblocky = div(data_size.y, BLOCK);
    auto nblockz = div(data_size.z, BLOCK);

    printf("\nSpline3::construct:\n");
    printf("ebx2: %lf\n", ebx2);
    printf("eb_r: %lf\n", eb_r);

    cusz::c_spline3d_infprecis_32x8x8data<TITER, EITER, float, 256, false>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
        (d_data, data_size, data_leap, d_ectrl, size_aligned, leap_aligned,
         d_anchor, anchor_leap, eb_r, ebx2, radius);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    return 0;
  }

  static int reconstruct(
      EITER in_ectrl, dim3 size_aligned, dim3 leap_aligned, TITER in_anchor,
      dim3 anchor_size, dim3 anchor_leap, double const eb, int const radius,
      TITER out_xdata, dim3 data_size, dim3 data_leap, cudaStream_t stream)
  {
    constexpr auto BLOCK = 8;
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

    auto nblockx = div(data_size.x, BLOCK * 4);
    auto nblocky = div(data_size.y, BLOCK);
    auto nblockz = div(data_size.z, BLOCK);

    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    cusz::x_spline3d_infprecis_32x8x8data<EITER, TITER, float, 256>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
        (in_ectrl, size_aligned, leap_aligned, in_anchor, anchor_size,
         anchor_leap, out_xdata, data_size, data_leap, eb_r, ebx2, radius);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    return 0;
  }

  // end of class definition
};

}  // namespace cusz

#endif
