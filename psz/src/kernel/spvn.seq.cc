#include "criteria.seq.hh"
#include "cusz/type.h"
#include "kernel/spv.hh"
#include "utils/timer.hh"

namespace psz {
namespace seq {

template <typename T, typename Criterion, typename M = u4>
void spvn_gather(
    T* in, szt const in_len, int const radius, T* cval, M* cidx, int* cn,
    Criterion criteria)
{
  for (auto tid = 0; tid < in_len; tid++) {
    auto d = in[tid];
    auto quantizable = criteria(d, radius);

    if (not quantizable) {
      auto cur_idx = (*cn)++;
      cidx[cur_idx] = tid;
      cval[cur_idx] = d;
      in[tid] = 0;
    }
  }
}

template <typename T, typename M = u4>
void spvn_scatter(T* val, M* idx, int const nnz, T* out)
{
  for (auto tid = 0; tid < nnz; tid++) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

}  // namespace seq
}  // namespace psz

#define SPVN_GATHER(T, Criterion, M)                                     \
  template <>                                                            \
  void psz::spv_gather_naive<SEQ, T, Criterion, M>(                      \
      T * in, size_t const in_len, int const radius, T* cval, M* cidx,   \
      int* cn, Criterion criteria, f4* milliseconds, void* stream)       \
  {                                                                      \
    psz::seq::spvn_gather(in, in_len, radius, cval, cidx, cn, criteria); \
  }

#define SPVN_SCATTER(T, M)                                       \
  template <>                                                    \
  void psz::spv_scatter_naive<SEQ, T, M>(                        \
      T * val, M * idx, int const nnz, T* out, f4* milliseconds, \
      void* stream)                                              \
  {                                                              \
    psz::seq::spvn_scatter(val, idx, nnz, out);                  \
  }

SPVN_SCATTER(f4, u4)
SPVN_SCATTER(f8, u4)

SPVN_GATHER(f4, psz::criterion::seq::eq<f4>, u4)
SPVN_GATHER(f4, psz::criterion::seq::in_ball<f4>, u4)
SPVN_GATHER(f4, psz::criterion::seq::in_ball_shifted<f4>, u4)
SPVN_GATHER(f8, psz::criterion::seq::eq<f8>, u4)
SPVN_GATHER(f8, psz::criterion::seq::in_ball<f8>, u4)
SPVN_GATHER(f8, psz::criterion::seq::in_ball_shifted<f8>, u4)

#undef SPVN_SCATTER