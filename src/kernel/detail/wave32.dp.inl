#ifndef C0001FA0_0D44_4EFB_BC50_2C6B3FE627F4
#define C0001FA0_0D44_4EFB_BC50_2C6B3FE627F4

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "warp_compat.dp.inl"

namespace psz::dpcpp::wave32 {

template <typename T, int SEQ, bool EXPERIMENTAL_MASKED = true>
__dpct_inline__ void intrawarp_inclscan_1d(
    T private_buffer[SEQ], const sycl::nd_item<3>& item_ct1)
{
  for (auto i = 1; i < SEQ; i++) private_buffer[i] += private_buffer[i - 1];
  T addend = private_buffer[SEQ - 1];

  // in-warp shuffle
  for (auto d = 1; d < 32; d *= 2) {
    T n = psz::dpcpp::compat::shift_sub_group_right(
        0xffffffff, item_ct1.get_sub_group(), addend, d);
    if (item_ct1.get_local_id(2) % 32 >= d) addend += n;
  }
  // exclusive scan
  T prev_addend = psz::dpcpp::compat::shift_sub_group_right(
      0xffffffff, item_ct1.get_sub_group(), addend, 1);

  // propagate
  if (item_ct1.get_local_id(2) % 32 > 0)
    for (auto i = 0; i < SEQ; i++) private_buffer[i] += prev_addend;
}

template <typename T, int SEQ, int NTHREAD>
__dpct_inline__ void intrablock_exclscan_1d(
    T private_buffer[SEQ], volatile T* exchange_in, volatile T* exchange_out,
    const sycl::nd_item<3>& item_ct1)
{
  constexpr auto NWARP = NTHREAD / 32;
  static_assert(NWARP <= 32, "too big");

  auto warp_id = item_ct1.get_local_id(2) / 32;
  auto lane_id = item_ct1.get_local_id(2) % 32;

  if (lane_id == 31) exchange_in[warp_id] = private_buffer[SEQ - 1];
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (NWARP <= 8) {
    if (item_ct1.get_local_id(2) == 0) {
      exchange_out[0] = 0;
      for (auto i = 1; i < NWARP; i++)
        exchange_out[i] = exchange_out[i - 1] + exchange_in[i - 1];
    }
  }
  else if (NWARP <= 32) {
    if (item_ct1.get_local_id(2) <= 32) {
      auto addend = exchange_in[item_ct1.get_local_id(2)];

      for (auto d = 1; d < 32; d *= 2) {
        T n = psz::dpcpp::compat::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), addend, d);
        if (item_ct1.get_local_id(2) >= d) addend += n;
      }
      // exclusive scan
      T prev_addend = /* DPCT1023 */
          psz::dpcpp::compat::shift_sub_group_right(
              0xffffffff, item_ct1.get_sub_group(), addend, 1);
      exchange_out[warp_id] = (warp_id > 0) * prev_addend;
    }
  }
  // else-case handled by static_assert
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // propagate
  auto addend = exchange_out[warp_id];
  for (auto i = 0; i < SEQ; i++) private_buffer[i] += addend;
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

}  // namespace psz::dpcpp::wave32

#endif /* C0001FA0_0D44_4EFB_BC50_2C6B3FE627F4 */
