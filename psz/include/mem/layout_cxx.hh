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
#include "memobj.hh"
#include "port.hh"

using namespace portable;

template <
    typename T, typename E, typename H, pszpolicy EXEC = PROPER_GPU_BACKEND>
class pszmempool_cxx {
 public:
  using M = uint32_t;
  using F = uint32_t;
  using B = uint8_t;
  using Compact = typename CompactDram<EXEC, T>::Compact;

  pszmem_cxx<T> *_oridata;              // original data
  pszmem_cxx<T> *_xdata, *_xdata_test;  // decomp'ed data (also for testing)
  pszmem_cxx<T> *_anchor;               // anchor
  pszmem_cxx<E> *_ectrl, *_ectrl_test;  // ectrl (_ectrl_test for testing)
  pszmem_cxx<F> *_hist;                 // hist/frequency

  Compact *compact;
  bool iscompression;

  pszmem_cxx<B> *_compressed;  // compressed

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
  F *hist() { return _hist->dptr(); }
  E *ectrl() { return _ectrl->dptr(); }
  T *anchor() { return _anchor->dptr(); }
  B *compressed() { return _compressed->dptr(); }
  B *compressed_h() { return _compressed->hptr(); }
  T *compact_val() { return compact->val(); }
  M *compact_idx() { return compact->idx(); }
  M compact_num_outliers() { return compact->num_outliers(); }
};

#define TPL template <typename T, typename E, typename H, pszpolicy EXEC>
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

  _compressed = new pszmem_cxx<B>(len * 1.2, 1, 1, "compressed");
  // _oridata = new pszmem_cxx<T>(x, y, z, "original data");
  if (not iscompression)
    _xdata = new pszmem_cxx<T>(x, y, z, "reconstructed data");
  // _xdata_test = new pszmem_cxx<T>(x, y, z, "reconstructed data (test)");
  _anchor = new pszmem_cxx<T>(div(x, BLK), div(y, BLK), div(z, BLK), "anchor");
  _ectrl = new pszmem_cxx<E>(x, y, z, "ectrl-lorenzo");
  // _ectrl_test = new pszmem_cxx<E>(x, y, z, "ectrl-lorenzo (test)");

  _hist = new pszmem_cxx<F>(bklen, 1, 1, "hist");

  _compressed->control({Malloc, MallocHost});
  _anchor->control({Malloc, MallocHost});
  _ectrl->control({Malloc, MallocHost});
  _hist->control({Malloc, MallocHost});

  if (iscompression) {
    // [psz::TODO] consider compact as a view with exposing the limited length
    // TODO not necessary for decompression
    compact = new Compact;
    compact->reserve_space(len / 5).control({Malloc, MallocHost});
  }
}

TPL POOL::~pszmempool_cxx()
{
  // cout << "entering pszmempool destructor" << endl;
  if (_anchor) delete _anchor;
  if (_ectrl) delete _ectrl;
  // if(_ectrl_test) delete _ectrl_test;
  if (_hist) delete _hist;
  // if(_oridata) delete _oridata;
  if (iscompression and _xdata) delete _xdata;
  // if(_xdata_test) delete _xdata_test;
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
