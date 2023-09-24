#include "detail/l23r.dp.inl"

#include <chrono>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "cusz/type.h"
#include "kernel/l23r.hh"
#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq, bool ZigZag>
pszerror psz_comp_l23r(
    T* const data, sycl::range<3> const len3, f8 const eb, int const radius,
    Eq* const eq, void* _outlier, f4* time_elapsed, void* stream)
{
  static_assert(
      std::is_same<Eq, u4>::value or std::is_same<Eq, uint16_t>::value or
          std::is_same<Eq, uint8_t>::value,
      "Eq must be unsigned integer that is less than or equal to 4 bytes.");

  auto divide3 = [](sycl::range<3> len, sycl::range<3> tile) {
    return sycl::range<3>(
        (len[2] - 1) / tile[2] + 1, (len[1] - 1) / tile[1] + 1,
        (len[0] - 1) / tile[0] + 1);
  };

  auto ndim = [&]() {
    if (len3[0] == 1 and len3[1] == 1)
      return 1;
    else if (len3[0] == 1 and len3[1] != 1)
      return 2;
    else
      return 3;
  };

  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  auto ot = (Compact*)_outlier;

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 4;
  constexpr auto Block1D = 64;
  auto Grid1D = divide3(len3, sycl::range<3>(1, 1, Tile1D));

  auto Tile2D = sycl::range<3>(1, 16, 16);
  auto Block2D = sycl::range<3>(1, 2, 16);
  auto Grid2D = divide3(len3, Tile2D);

  auto Tile3D = sycl::range<3>(8, 8, 32);
  auto Block3D = sycl::range<3>(1, 8, 32);
  auto Grid3D = divide3(len3, Tile3D);

  auto d = ndim();

  auto queue = (sycl::queue*)((dpct::queue_ptr)stream);

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = sycl::range<3>(len3[2] * len3[1], len3[2], 1);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(queue);

  if (d == 1) {
    /*
    DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    queue->submit([&](sycl::handler& cgh) {
      using EqUint = typename psz::typing::UInt<sizeof(Eq)>::T;
      using EqInt = typename psz::typing::Int<sizeof(Eq)>::T;

      sycl::local_accessor<T, 1> s_data(sycl::range<1>(Tile1D), cgh);
      sycl::local_accessor<EqUint, 1> s_eq(sycl::range<1>(Tile1D), cgh);

      auto ot_val_ct6 = ot->val();
      auto ot_idx_ct7 = ot->idx();
      auto ot_num_ct8 = ot->num();

      cgh.parallel_for(
          sycl::nd_range<3>(
              Grid1D * sycl::range<3>(1, 1, Block1D),
              sycl::range<3>(1, 1, Block1D)),
          [=](sycl::nd_item<3> item_ct1) {
            psz::rolling_dp::c_lorenzo_1d1l<T, false, Eq, T, Tile1D, Seq1D>(
                data, len3, leap3, radius, ebx2_r, eq, ot_val_ct6, ot_idx_ct7,
                ot_num_ct8, item_ct1, s_data.get_pointer(),
                s_eq.get_pointer());
          });
    });
  }
  else if (d == 2) {
    /*
    DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    queue->submit([&](sycl::handler& cgh) {
      auto ot_val_ct6 = ot->val();
      auto ot_idx_ct7 = ot->idx();
      auto ot_num_ct8 = ot->num();

      cgh.parallel_for(
          sycl::nd_range<3>(Grid2D * Block2D, Block2D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::rolling_dp::c_lorenzo_2d1l<T, false, Eq, T>(
                data, len3, leap3, radius, ebx2_r, eq, ot_val_ct6, ot_idx_ct7,
                ot_num_ct8, item_ct1);
          });
    });
  }
  else if (d == 3) {
    /*
    DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    queue->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<T, 2> s_acc_ct1(sycl::range<2>(9, 33), cgh);

      auto ot_val_ct6 = ot->val();
      auto ot_idx_ct7 = ot->idx();
      auto ot_num_ct8 = ot->num();

      cgh.parallel_for(
          sycl::nd_range<3>(Grid3D * Block3D, Block3D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::rolling_dp::c_lorenzo_3d1l<T, false, Eq, T>(
                data, len3, leap3, radius, ebx2_r, eq, ot_val_ct6, ot_idx_ct7,
                ot_num_ct8, item_ct1, s_acc_ct1);
          });
    });
  }

  STOP_GPUEVENT_RECORDING(queue);
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#define INIT(T, E, ZIGZAG)                                             \
  template pszerror psz_comp_l23r<T, E, ZIGZAG>(                       \
      T* const data, sycl::range<3> const len3, f8 const eb,           \
      int const radius, E* const eq, void* _outlier, f4* time_elapsed, \
      void* queue);

INIT(f4, u4, false)
// INIT(f4, u4, true)
// INIT(f8, u4, false)
// INIT(f8, u4, true)

#undef INIT