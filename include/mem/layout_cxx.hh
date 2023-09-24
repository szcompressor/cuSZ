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
#include "layout.h"
#include "memseg_cxx.hh"
#include "port.hh"

template <typename T, typename E, typename H>
class pszmempool_cxx {
 public:
  using M = uint32_t;
  using F = uint32_t;
  using B = uint8_t;
  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;

  pszmem_cxx<T> *od;  // original data
  pszmem_cxx<T> *xd;  // decompressed/reconstructed data
  pszmem_cxx<T> *ac;  // anchor

  pszmem_cxx<E> *e, *el, *es;  // ectrl, e-Lorenzo, e-spline
  pszmem_cxx<F> *ht;           // histogram/frequency

  pszmem_cxx<T> *oc;  // outlier compat
  pszmem_cxx<T> *sv;  // sp-val
  pszmem_cxx<M> *si;  // sp-idx

  Compact *compact;

  pszmem_cxx<B> *_compressed;  // compressed

  size_t len, len_spl;
  int radius, bklen;

 public:
  // ctor, dtor
  pszmempool_cxx(u4 _x, int _radius, u4 _y, u4 _z);
  ~pszmempool_cxx();
  // utils
  pszmempool_cxx *clear_buffer();
  // getter
  T *outlier_space() { return oc->dptr(); }
  T *outlier_val() { return sv->dptr(); }
  M *outlier_idx() { return si->dptr(); }
  F *hist() { return ht->dptr(); }
  E *ectrl_lrz() { return el->dptr(); }
  E *ectrl_spl() { return es->dptr(); }
  T *anchor() { return ac->dptr(); }
  B *compressed() { return _compressed->dptr(); }
  B *compressed_h() { return _compressed->hptr(); }
  T *compact_val() { return compact->val(); }
  M *compact_idx() { return compact->idx(); }
  M compact_num_outliers() { return compact->num_outliers(); }
};

#define TPL template <typename T, typename E, typename H>
#define POOL pszmempool_cxx<T, E, H>

TPL POOL::pszmempool_cxx(u4 x, int _radius, u4 y, u4 z)
{
  len = x * y * z;
  radius = _radius;
  bklen = 2 * radius;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto pad = [&](auto _l, auto unit) { return unit * div(_l, unit); };

  // for spline
  constexpr auto BLK = 8;
  auto xp = pad(x, 4 * BLK), yp = (y == 1) ? 1 : pad(y, BLK),
       zp = (z == 1) ? 1 : pad(z, BLK);
  len_spl = xp * yp * zp;  // always larger than len

  _compressed = new pszmem_cxx<B>(len * 1.2, 1, 1, "compressed");

  od = new pszmem_cxx<T>(x, y, z, "original data");
  xd = new pszmem_cxx<T>(x, y, z, "reconstructed data");
  ac = new pszmem_cxx<T>(div(x, BLK), div(y, BLK), div(z, BLK), "anchor");

  oc = new pszmem_cxx<T>(x, y, z, "outlier space, compat");

  e = new pszmem_cxx<E>(len_spl, 1, 1, "ectrl-space");
  el = new pszmem_cxx<E>(x, y, z, "ectrl-lorenzo");
  es = new pszmem_cxx<E>(xp, yp, zp, "ectrl-spline");

  ht = new pszmem_cxx<F>(bklen, 1, 1, "hist");
  // sv = new pszmem_cxx<T>(x, y, z, "sp-val");
  // si = new pszmem_cxx<M>(x, y, z, "sp-idx");

  compact = new Compact;

  _compressed->control({Malloc, MallocHost});
  oc->control({Malloc, MallocHost});
  ac->control({Malloc, MallocHost});
  e->control({Malloc, MallocHost});
  ht->control({Malloc, MallocHost});
  // sv->control({Malloc, MallocHost});
  // si->control({Malloc, MallocHost});

  compact->reserve_space(len / 5).control({Malloc, MallocHost});

  el->asaviewof(e);
  es->asaviewof(e);
}

TPL POOL::~pszmempool_cxx()
{
  delete ac, delete e, delete oc, delete ht, delete sv, delete si;
  compact->control({Free, FreeHost});
}

TPL POOL *POOL::clear_buffer()
{
  e->control({ClearDevice});
  ac->control({ClearDevice});
  oc->control({ClearDevice});
  // sv->control({ClearDevice});
  // si->control({ClearDevice});
  _compressed->control({ClearDevice});

  delete compact;

  return this;
}

#undef TPL
#undef POOL

#endif /* DC62DA60_8211_4C93_9541_950ADEFC2820 */
