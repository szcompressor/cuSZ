/**
 * @file hist_sp.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

//                    -2 -1  0 +1 +2
//                     |<-R->|<-R->|
// p_hist              |<----K---->|       K=2R+1
// s_hist |<-----offset------|------offset------>|
//
// Multiple warps:the large shmem use (relative to #thread).

template <
    typename T, typename FQ = uint32_t, int K = 5,
    bool OneapiUseExperimental = true>
void histsp_multiwarp(
    T *in, uint32_t inlen,  //
    uint32_t chunk, FQ *out, uint32_t outlen, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local, int offset = 0)
{
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;  // K = 5, R = 2

  // small & big local hist based on presumed access category
  auto s_hist = (FQ *)dpct_local;
  // cannot scale according to compressiblity becasuse of the register pressure
  // there should be offline optimal
  FQ p_hist[K] = {0};

  auto global_id = [&](auto i, const sycl::nd_item<3> &item_ct1) {
    return item_ct1.get_group(2) * chunk + i;
  };
  auto nworker = [&](const sycl::nd_item<3> &item_ct1) {
    return item_ct1.get_local_range(2);
  };

  for (auto i = item_ct1.get_local_id(2); i < outlen;
       i += item_ct1.get_local_range(2))
    s_hist[i] = 0;
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (auto i = item_ct1.get_local_id(2); i < chunk;
       i += item_ct1.get_local_range(2)) {
    auto gid = global_id(i, item_ct1);
    if (gid < inlen) {
      auto ori = (int)in[gid];  // e.g., 512
      auto sym = ori - offset;  // e.g., 512 - 512 = 0

      if (2 * sycl::abs(sym) < K) {
        // -2, -1, 0, 1, 2 -> sym + R = 0, 1, 2, 3, 4
        //  4   2  0  2  4 <- 2 * abs(sym)
        p_hist[sym + R] += 1;  // more possible
      }
      else {
        // resume the original input
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &s_hist[ori], 1);  // less possible
      }
    }
  }
  /* DPCT1065 */
  // No global write, is local_space okay?
  // item_ct1.barrier();
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (auto &sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      /* DPCT1108 */
      if constexpr (OneapiUseExperimental) {
        auto n = dpct::experimental::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), sum, d);
        if (item_ct1.get_local_id(2) % 32 >= d) sum += n;
      }
      else {
        /* DPCT1023 */
        /* DPCT1096 */
        auto n = dpct::shift_sub_group_right(item_ct1.get_sub_group(), sum, d);
        if (item_ct1.get_local_id(2) % 32 >= d) sum += n;
      }
    }
  }

  for (auto i = 0; i < K; i++)
    if (item_ct1.get_local_id(2) % 32 == 31)
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &s_hist[(int)offset + i - R], p_hist[i]);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (auto i = item_ct1.get_local_id(2); i < outlen;
       i += item_ct1.get_local_range(2))
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out + i, s_hist[i]);
}
