/**
 * @file layout_cxx.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-09
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DC62DA60_8211_4C93_9541_950ADEFC2820
#define DC62DA60_8211_4C93_9541_950ADEFC2820

#include "compact.hh"
#include "cusz/type.h"
#include "layout.h"
#include "mem/definition.hh"
#include "memobj.hh"
#include "port.hh"

using namespace portable;

template <
    typename T, typename E, typename H, psz_policy EXEC = PROPER_GPU_BACKEND>
class pszmempool_cxx {
 public:
  using M = uint32_t;
  using F = uint32_t;
  using B = uint8_t;
  using Compact = typename CompactDram<EXEC, T>::Compact;

  memobj<T> *_oridata;              // original data
  memobj<T> *_anchor;               // anchor
  memobj<E> *_ectrl, *_ectrl_test;  // ectrl (_ectrl_test for testing)
  memobj<F> *_hist;                 // hist/frequency

  Compact *compact;
  bool iscompression;

  memobj<B> *_compressed;  // compressed

  size_t len;
  int radius, bklen;

 public:
  // ctor, dtor
  pszmempool_cxx(
      u4 _x, int _radius = 32768, u4 _y = 1, u4 _z = 1,
      bool iscompression = true);
  ~pszmempool_cxx();
  // utils
  pszmempool_cxx *clear_buffer();
  // getter
  F *hist() const { return _hist->dptr(); }
  E *ectrl() const { return _ectrl->dptr(); }
  T *anchor() const { return _anchor->dptr(); }
  B *compressed() const { return _compressed->dptr(); }
  B *compressed_h() const { return _compressed->hptr(); }
  T *compact_val() const { return compact->val(); }
  M *compact_idx() const { return compact->idx(); }
  M compact_num_outliers() const { return compact->num_outliers(); }
  compact_array1<T> outlier()
  {
    return compact_array1<T>{
        compact->d_val, compact->d_idx, compact->d_num, compact->h_num,
        compact->reserved_len};
  }
};

#define TPL template <typename T, typename E, typename H, psz_policy EXEC>
#define POOL pszmempool_cxx<T, E, H, EXEC>

TPL POOL::pszmempool_cxx(u4 x, int _radius, u4 y, u4 z, bool _iscompression) :
    iscompression(_iscompression)
{
  len = x * y * z;
  radius = _radius;
  bklen = 2 * radius;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto pad = [&](auto _l, auto unit) { return unit * div(_l, unit); };

  // for spline
  constexpr auto BLK = 8;

  _compressed =
      new memobj<B>(len * 4 / 2, "psz::comp'ed", {Malloc, MallocHost});
  _anchor = new memobj<T>(
      div(x, BLK), div(y, BLK), div(z, BLK), "psz::anchor",
      {Malloc, MallocHost});
  _ectrl = new memobj<E>(x, y, z, "psz::quant", {Malloc});
  _hist = new memobj<F>(bklen, "psz::hist", {Malloc, MallocHost});

  if (iscompression) {
    // [psz::TODO] consider compact as a view with exposing the limited length
    compact = new Compact(len / 5);
    compact->control({Malloc, MallocHost});
  }
}

TPL POOL::~pszmempool_cxx()
{
  if (_anchor) delete _anchor;
  if (_ectrl) delete _ectrl;
  if (_hist) delete _hist;
  if (_compressed) delete _compressed;

  if (iscompression) {
    compact->control({Free, FreeHost});
    delete compact;
  }
}

TPL POOL *POOL::clear_buffer()
{
  _ectrl->control({ClearDevice});
  _anchor->control({ClearDevice});
  _compressed->control({ClearDevice});

  delete compact;

  return this;
}

#undef TPL
#undef POOL

#endif /* DC62DA60_8211_4C93_9541_950ADEFC2820 */
