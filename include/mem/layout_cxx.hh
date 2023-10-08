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

#include "cusz/type.h"
#include "compact.hh"
#include "layout.h"
#include "memseg_cxx.hh"
#include "port.hh"

template <
    typename T, typename E, typename H, pszpolicy EXEC = PROPER_GPU_BACKEND>
class pszmempool_cxx {
 public:
  using M = uint32_t;
  using F = uint32_t;
  using B = uint8_t;
  using Compact = typename CompactDram<EXEC, T>::Compact;

  pszmem_cxx<T> *od;  // original data
  pszmem_cxx<T> *xd;  // decompressed/reconstructed data
  pszmem_cxx<T> *ac;  // anchor
  pszmem_cxx<E> *e;   // ectrl
  pszmem_cxx<F> *ht;  // hist/frequency

  Compact *compact;

  pszmem_cxx<B> *_compressed;  // compressed

  size_t len;
  int radius, bklen;

 public:
  // ctor, dtor
  pszmempool_cxx(
      u4 _x, int _radius = 32768, u4 _y = 1, u4 _z = 1);  // ori radius 512
  ~pszmempool_cxx();
  // utils
  pszmempool_cxx *clear_buffer();
  // getter
  F *hist() { return ht->dptr(); }
  E *ectrl() { return e->dptr(); }
  T *anchor() { return ac->dptr(); }
  B *compressed() { return _compressed->dptr(); }
  B *compressed_h() { return _compressed->hptr(); }
  T *compact_val() { return compact->val(); }
  M *compact_idx() { return compact->idx(); }
  M compact_num_outliers() { return compact->num_outliers(); }
};

#define TPL template <typename T, typename E, typename H, pszpolicy EXEC>
#define POOL pszmempool_cxx<T, E, H, EXEC>

TPL POOL::pszmempool_cxx(u4 x, int _radius, u4 y, u4 z)
{
  len = x * y * z;
  radius = _radius;
  bklen = 2 * radius;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto pad = [&](auto _l, auto unit) { return unit * div(_l, unit); };

  // for spline
  constexpr auto BLK = 8;

  _compressed = new pszmem_cxx<B>(len * 1.2, 1, 1, "compressed");

  od = new pszmem_cxx<T>(x, y, z, "original data");
  xd = new pszmem_cxx<T>(x, y, z, "reconstructed data");
  ac = new pszmem_cxx<T>(div(x, BLK), div(y, BLK), div(z, BLK), "anchor");
  e = new pszmem_cxx<E>(x, y, z, "ectrl-lorenzo");

  ht = new pszmem_cxx<F>(bklen, 1, 1, "hist");

  compact = new Compact;

  _compressed->control({Malloc, MallocHost});
  ac->control({Malloc, MallocHost});
  e->control({Malloc, MallocHost});
  ht->control({Malloc, MallocHost});

  // [psz::TODO] consider compact as a view with exposing the limited length
  compact->reserve_space(len / 5).control({Malloc, MallocHost});
}

TPL POOL::~pszmempool_cxx()
{
  delete ac, delete e, delete ht;
  compact->control({Free, FreeHost});
}

TPL POOL *POOL::clear_buffer()
{
  e->control({ClearDevice});
  ac->control({ClearDevice});
  _compressed->control({ClearDevice});

  delete compact;

  return this;
}

#undef TPL
#undef POOL

#endif /* DC62DA60_8211_4C93_9541_950ADEFC2820 */
