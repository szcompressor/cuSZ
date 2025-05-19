#include "mem/buf_comp.hh"

#include "cusz/type.h"

template <typename T, typename E>
struct psz::Buf_Comp<T, E>::impl {
  const psz_len len;
  const size_t len_linear;
  const size_t len_linear_anchor;  // for spline

  // state
  bool is_comp;

  // arrays
  GPU_unique_dptr<E[]> d_ectrl;
  GPU_unique_dptr<BYTE[]> d_compressed;
  GPU_unique_hptr<BYTE[]> h_compressed;
  GPU_unique_dptr<Freq[]> d_hist;
  GPU_unique_hptr<Freq[]> h_hist;
  GPU_unique_dptr<Freq[]> d_top1;
  GPU_unique_hptr<Freq[]> h_top1;

  std::unique_ptr<Buf_Outlier> buf_outlier;
  std::unique_ptr<Buf_Outlier2> buf_outlier2;
  std::unique_ptr<Buf_HF> buf_hf;

  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

  // spline-specific: declare
  GPU_unique_dptr<T[]> d_anchor;
  GPU_unique_dptr<T[]> d_pe;
  GPU_unique_hptr<T[]> h_pe;

 private:
  static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

  static size_t set_anchor_len(u4 x, u4 y, u4 z)
  {
    return _div(x, BLK) * _div(y, BLK) * _div(z, BLK);
  }

  static size_t set_anchor_len(psz_len len)
  {
    return _div(len.x, BLK) * _div(len.y, BLK) * _div(len.z, BLK);
  }

 public:
  impl(psz_len _len, BufToggle_Comp* toggle) :
      len(_len), len_linear(_len.x * _len.y * _len.z), len_linear_anchor(set_anchor_len(_len))
  {
    if (toggle->use_quant) d_ectrl = MAKE_UNIQUE_DEVICE(E, len_linear);
    if (toggle->use_outlier) {
      buf_outlier = std::make_unique<Buf_Outlier>(len_linear * OUTLIER_RATIO);
      buf_outlier2 = std::make_unique<Buf_Outlier2>(len_linear * OUTLIER_RATIO);
    }
    if (toggle->use_anchor) d_anchor = MAKE_UNIQUE_DEVICE(T, len_linear_anchor);
    if (toggle->use_hist) {
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
    }
    if (toggle->use_compressed) {
      d_compressed = MAKE_UNIQUE_DEVICE(BYTE, len_linear * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(BYTE, len_linear * 4 / 2);
    }
    if (toggle->use_top1) {
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, 1);
    }
  }

  impl(psz_len _len, bool _is_comp) :
      is_comp(_is_comp),
      len(_len),
      len_linear(_len.x * _len.y * _len.z),
      len_linear_anchor(set_anchor_len(_len))
  {
    // align 4Ki for (essentially) FZG
    d_ectrl = MAKE_UNIQUE_DEVICE(E, ALIGN_4Ki(len_linear));

    if (is_comp) {
      d_anchor = MAKE_UNIQUE_DEVICE(T, len_linear_anchor);
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
      d_compressed = MAKE_UNIQUE_DEVICE(BYTE, len_linear * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(BYTE, len_linear * 4 / 2);
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, 1);

      buf_outlier = std::make_unique<Buf_Outlier>(len_linear * OUTLIER_RATIO);
      buf_outlier2 = std::make_unique<Buf_Outlier2>(len_linear * OUTLIER_RATIO);
      buf_hf = std::make_unique<Buf_HF>(len_linear, max_bklen);
    }
  }

  ~impl() {};

  void clear_buffer()
  {
    memset_device(d_ectrl.get(), len_linear);
    memset_device(d_hist.get(), max_bklen);
    memset_device(d_anchor.get(), len_linear_anchor);
    memset_device(d_compressed.get(), len_linear * 4 / 2);
    // TODO clear buf_outlier
  }
};

#define COMPBUF_IMPL(RET_TYPE)      \
  template <typename T, typename E> \
  RET_TYPE Buf_Comp<T, E>

namespace psz {

COMPBUF_IMPL()::Buf_Comp(psz_len _len, BufToggle_Comp* toggle) :
    len(_len), len_linear(_len.x * _len.y * _len.z), pimpl(std::make_unique<impl>(_len, toggle))
{
}

COMPBUF_IMPL()::Buf_Comp(psz_len _len, bool _is_comp) :
    is_comp(_is_comp),
    len(_len),
    len_linear(_len.x * _len.y * _len.z),
    pimpl(std::make_unique<impl>(_len, _is_comp))
{
}

COMPBUF_IMPL()::~Buf_Comp(){};

COMPBUF_IMPL(void)::clear_buffer() { pimpl->clear_buffer(); }

COMPBUF_IMPL(void)::clear_top1() { memset_device(pimpl->d_top1.get(), 1); }

// getters: array
COMPBUF_IMPL(E*)::ectrl_d() const { return pimpl->d_ectrl.get(); }
COMPBUF_IMPL(psz_len)::ectrl_len3() const { return len; }
COMPBUF_IMPL(E*)::eq_d() const { return pimpl->d_ectrl.get(); }
COMPBUF_IMPL(psz_len)::eq_len3() const { return len; }

COMPBUF_IMPL(Freq*)::hist_d() const { return pimpl->d_hist.get(); }
COMPBUF_IMPL(Freq*)::hist_h() const { return pimpl->h_hist.get(); }

COMPBUF_IMPL(Freq*)::top1_d() const { return pimpl->d_top1.get(); }
COMPBUF_IMPL(Freq*)::top1_h() const
{
  memcpy_allkinds<D2H>(pimpl->h_top1.get(), pimpl->d_top1.get(), 1);
  return pimpl->h_top1.get();
}

COMPBUF_IMPL(BYTE*)::compressed_d() const { return pimpl->d_compressed.get(); }
COMPBUF_IMPL(BYTE*)::compressed_h() const { return pimpl->h_compressed.get(); }

COMPBUF_IMPL(T*)::outlier_val_d() const { return pimpl->buf_outlier->val(); }
COMPBUF_IMPL(M*)::outlier_idx_d() const { return pimpl->buf_outlier->idx(); }
COMPBUF_IMPL(M)::outlier_num() const { return pimpl->buf_outlier->num_outliers(); }

COMPBUF_IMPL(void*)::outlier2_validx_d() const { return pimpl->buf_outlier2->val_idx_d(); }
COMPBUF_IMPL(M)::outlier2_host_get_num() const { return pimpl->buf_outlier2->host_get_num(); }

COMPBUF_IMPL(T*)::anchor_d() const { return pimpl->d_anchor.get(); }
COMPBUF_IMPL(size_t)::anchor_len() const { return pimpl->len_linear_anchor; }
COMPBUF_IMPL(psz_len)::anchor_len3() const
{
  auto _div = [](size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };
  return {_div(len.x, BLK), _div(len.y, BLK), _div(len.z, BLK)};
}

COMPBUF_IMPL(T*)::profiled_errors_d() const { return pimpl->d_pe.get(); };
COMPBUF_IMPL(T*)::profiled_errors_h() const { return pimpl->h_pe.get(); };
COMPBUF_IMPL(M)::profiled_errors_len() const { return ERR_HISTO_LEN; };

template <typename T>
using Buf_Outlier = _portable::compact_gpu<T>;

template <typename T>
using Buf_Outlier2 = _portable::compact_GPU_DRAM2<T, M>;

template <typename E>
using Buf_HF = phf::Buf<E>;

COMPBUF_IMPL(Buf_Outlier<T>*)::buf_outlier() const { return pimpl->buf_outlier.get(); }
COMPBUF_IMPL(Buf_Outlier2<T>*)::buf_outlier2() const { return pimpl->buf_outlier2.get(); }

COMPBUF_IMPL(Buf_HF<E>*)::buf_hf() const { return pimpl->buf_hf.get(); }

}  // namespace psz

// instantiation
template class psz::Buf_Comp<f4, u1>;
template class psz::Buf_Comp<f4, u2>;
template class psz::Buf_Comp<f8, u1>;
template class psz::Buf_Comp<f8, u2>;
