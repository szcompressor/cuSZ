#include "detail/spvn.dp.inl"

#include "criteria.gpu.hh"
#include "cusz/type.h"
#include "kernel/spv.hh"
#include "port.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define SPVN_GATHER(T, Criterion, M)                                        \
  template <>                                                               \
  void psz::spv_gather_naive<ONEAPI, T, Criterion, M>(                      \
      T * in, size_t const in_len, int const radius, T* cval, M* cidx,      \
      int* cn, Criterion criteria, f4* milliseconds, void* stream)          \
  {                                                                         \
    auto q = (sycl::queue*)stream;                                          \
    auto grid_dim = (in_len - 1) / 128 + 1;                                 \
    sycl::event e = q->submit([&](sycl::handler& cgh) {                     \
      auto in_ct0 = in;                                                     \
      auto in_len_ct1 = in_len;                                             \
      auto radius_ct2 = radius;                                             \
      auto cval_ct3 = cval;                                                 \
      auto cidx_ct4 = cidx;                                                 \
      auto cn_ct5 = cn;                                                     \
      auto criteria_ct0 = criteria;                                         \
                                                                            \
      cgh.parallel_for(                                                     \
          sycl::nd_range<3>(                                                \
              sycl::range<3>(1, 1, grid_dim) * sycl::range<3>(1, 1, 128),   \
              sycl::range<3>(1, 1, 128)),                                   \
          [=](sycl::nd_item<3> item_ct1) {                                  \
            psz::dpcpp::spvn_gather(                                        \
                in_ct0, in_len_ct1, radius_ct2, cval_ct3, cidx_ct4, cn_ct5, \
                criteria_ct0, item_ct1);                                    \
          });                                                               \
    });                                                                     \
    e.wait();                                                               \
    SYCL_TIME_DELTA(e, *milliseconds);                                      \
  }

#define SPVN_SCATTER(T, M)                                                \
  template <>                                                             \
  void psz::spv_scatter_naive<ONEAPI, T, M>(                              \
      T * val, M * idx, int const nnz, T* out, f4* milliseconds,          \
      void* stream)                                                       \
  {                                                                       \
    auto grid_dim = (nnz - 1) / 128 + 1;                                  \
    auto q = (sycl::queue*)stream;                                        \
    sycl::event e = q->submit([&](sycl::handler& cgh) {                   \
      auto val_ct0 = val;                                                 \
      auto idx_ct1 = idx;                                                 \
      auto nnz_ct2 = nnz;                                                 \
      auto out_ct3 = out;                                                 \
                                                                          \
      cgh.parallel_for(                                                   \
          sycl::nd_range<3>(                                              \
              sycl::range<3>(1, 1, grid_dim) * sycl::range<3>(1, 1, 128), \
              sycl::range<3>(1, 1, 128)),                                 \
          [=](sycl::nd_item<3> item_ct1) {                                \
            psz::dpcpp::spvn_scatter<T, M>(                               \
                val_ct0, idx_ct1, nnz_ct2, out_ct3, item_ct1);            \
          });                                                             \
    });                                                                   \
    e.wait();                                                             \
    SYCL_TIME_DELTA(e, *milliseconds);                                    \
  }

/* DPCT1012 */
SPVN_SCATTER(f4, u4)
// SPVN_SCATTER(f8, u4)

SPVN_GATHER(f4, psz::criterion::gpu::eq<f4>, u4)
SPVN_GATHER(f4, psz::criterion::gpu::in_ball<f4>, u4)
SPVN_GATHER(f4, psz::criterion::gpu::in_ball_shifted<f4>, u4)
// SPVN_GATHER(f8, psz::criterion::eq<f8>, u4)
// SPVN_GATHER(f8, psz::criterion::in_ball<f8>, u4)
// SPVN_GATHER(f8, psz::criterion::in_ball_shifted<f8>, u4)

#undef SPVN_SCATTER