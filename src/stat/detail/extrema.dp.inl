/**
 * @file extrema_g.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-19
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760
#define E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760

#include <math.h>
#include <stdio.h>

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace psz {

namespace {

template <int bytewidth>
struct matchby;
template <>
struct matchby<4> {
  using utype = unsigned int;
  using itype = int;
  using ftype = float;
};
template <>
struct matchby<8> {
  using utype = unsigned long long;
  using itype = long long;
  using ftype = double;
};

#define __ATOMIC_PLUGIN                                                     \
  constexpr auto bytewidth = sizeof(T);                                     \
  using itype = typename matchby<bytewidth>::itype;                         \
  using utype = typename matchby<bytewidth>::utype;                         \
  using ftype = typename matchby<bytewidth>::ftype;                         \
  static_assert(std::is_same<T, ftype>::value, "T and ftype don't match."); \
  auto fp_as_int = [](T fpval) -> itype {                                   \
    return *reinterpret_cast<itype *>(&fpval);                              \
  };                                                                        \
  auto fp_as_uint = [](T fpval) -> utype {                                  \
    return *reinterpret_cast<utype *>(&fpval);                              \
  };                                                                        \
  auto int_as_fp = [](itype ival) -> T {                                    \
    return *reinterpret_cast<T *>(&ival);                                   \
  };                                                                        \
  auto uint_as_fp = [](utype uval) -> T {                                   \
    return *reinterpret_cast<T *>(&uval);                                   \
  };

// modifed from https://stackoverflow.com/a/51549250 (CC BY-SA 4.0)
// https://stackoverflow.com/a/72461459
template <typename T>
__dpct_inline__ T atomicMinFp(T *addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !sycl::signbit(value)
                 ? int_as_fp(dpct::atomic_fetch_min<
                             sycl::access::address_space::generic_space>(
                       (itype *)addr, fp_as_int(value)))
                 : uint_as_fp(dpct::atomic_fetch_max<
                              sycl::access::address_space::generic_space>(
                       (utype *)addr, fp_as_uint(value)));
  return old;
}

template <typename T>
__dpct_inline__ T atomicMaxFp(T *addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !sycl::signbit(value)
                 ? int_as_fp(dpct::atomic_fetch_max<
                             sycl::access::address_space::generic_space>(
                       (itype *)addr, fp_as_int(value)))
                 : uint_as_fp(dpct::atomic_fetch_min<
                              sycl::access::address_space::generic_space>(
                       (utype *)addr, fp_as_uint(value)));
  return old;
}

}  // namespace

template <typename T>
void extrema_kernel(
    T *in, size_t const len, T *minel, T *maxel, T const failsafe, int const R,
    const sycl::nd_item<3> &item_ct1, T &shared_minv, T &shared_maxv)
{
  T tp_minv, tp_maxv;

  auto entry = (item_ct1.get_local_range(2) * R) * item_ct1.get_group(2) +
               item_ct1.get_local_id(2);
  auto _idx = [&](auto r, const sycl::nd_item<3> &item_ct1) {
    return entry + (r * item_ct1.get_local_range(2));
  };

  // failsafe; require external setup
  tp_minv = failsafe, tp_maxv = failsafe;
  if (item_ct1.get_local_id(2) == 0)
    shared_minv = failsafe, shared_maxv = failsafe;

  /*
  DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for (auto r = 0; r < R; r++) {
    auto idx = _idx(r, item_ct1);
    if (idx < len) {
      auto val = in[idx];

      tp_minv = dpct::min(tp_minv, val);
      tp_maxv = dpct::max(tp_maxv, val);
    }
  }
  /*
  DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  atomicMinFp<T>(&shared_minv, tp_minv);
  atomicMaxFp<T>(&shared_maxv, tp_maxv);
  /*
  DPCT1065:28: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    auto oldmin = atomicMinFp<T>(minel, shared_minv);
    auto oldmax = atomicMaxFp<T>(maxel, shared_maxv);
  }
}

}  // namespace psz

namespace psz {

namespace dpcpp {

template <typename T>
void extrema(T *in, size_t len, T res[4])
{
  // [TODO] external stream
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &queue = dev_ct1.default_queue();

  static const int MINVAL = 0;
  static const int MAXVAL = 1;
  //   static const int AVGVAL = 2;  // TODO
  static const int RNG = 3;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto chunk = 32768;
  auto nworker = 128;
  auto R = chunk / nworker;

  T h_min, h_max, failsafe;
  T *d_minel, *d_maxel;

  d_minel = (T *)sycl::malloc_device(sizeof(T), queue);
  d_maxel = (T *)sycl::malloc_device(sizeof(T), queue);

  // failsafe init
  queue.memcpy(&failsafe, in, sizeof(T)).wait();
  queue.memcpy(d_minel, in, sizeof(T)).wait();
  queue.memcpy(d_maxel, in, sizeof(T)).wait();

  /*
  DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    // dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 0> shared_minv_acc_ct1(cgh);
      sycl::local_accessor<T, 0> shared_maxv_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              div(len, chunk) * sycl::range<3>(1, 1, nworker),
              sycl::range<3>(1, 1, nworker)),
          [=](sycl::nd_item<3> item_ct1) {
            psz::extrema_kernel<T>(
                in, len, d_minel, d_maxel, failsafe, R, item_ct1,
                shared_minv_acc_ct1, shared_maxv_acc_ct1);
          });
    });
  }

  queue.wait();

  // collect results
  queue.memcpy(&h_min, d_minel, sizeof(T)).wait();
  queue.memcpy(&h_max, d_maxel, sizeof(T)).wait();

  res[MINVAL] = h_min;
  res[MAXVAL] = h_max;
  res[RNG] = h_max - h_min;

  sycl::free(d_minel, queue);
  sycl::free(d_maxel, queue);

  // dev_ct1.destroy_queue(stream);
}

}  // namespace dpcpp
}  // namespace psz

#endif /* E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760 */
