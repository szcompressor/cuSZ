#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace psz {
namespace cu_hip {
namespace wave32 {

template <typename T, int SEQ>
__dpct_inline__ void intrawarp_inclscan_1d(
    T private_buffer[SEQ], const sycl::nd_item<3>& item_ct1)
{
  for (auto i = 1; i < SEQ; i++) private_buffer[i] += private_buffer[i - 1];
  T addend = private_buffer[SEQ - 1];

  // in-warp shuffle
  for (auto d = 1; d < 32; d *= 2) {
    /*
    DPCT1023: 15: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_right. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_up_sync.
    */
    T n = dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, d);
    if (item_ct1.get_local_id(2) % 32 >= d) addend += n;
  }
  // exclusive scan
  /*
  DPCT1023:16: The SYCL sub-group does not support mask options for
  dpct::shift_sub_group_right. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_up_sync.
  */
  T prev_addend =
      dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, 1);

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
        /*
        DPCT1023:20: The SYCL sub-group does not support mask options for
        dpct::shift_sub_group_right. You can specify
        "--use-experimental-features=masked-sub-group-operation" to use the
        experimental helper function to migrate __shfl_up_sync.
        */
        T n = dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, d);
        if (item_ct1.get_local_id(2) >= d) addend += n;
      }
      // exclusive scan
      /*
      DPCT1023:21: The SYCL sub-group does not support mask options for
      dpct::shift_sub_group_right. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_up_sync.
      */
      T prev_addend =
          dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, 1);
      exchange_out[warp_id] = (warp_id > 0) * prev_addend;
    }
  }
  // else-case handled by static_assert
  /*
  DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // propagate
  auto addend = exchange_out[warp_id];
  for (auto i = 0; i < SEQ; i++) private_buffer[i] += addend;
  /*
  DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
}

}  // namespace wave32
}  // namespace cu_hip
}  // namespace psz