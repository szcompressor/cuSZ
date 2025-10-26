#include "mem/buf_comp.hh"

template <typename T, typename E>
struct psz::Buf_Comp<T, E>::impl {
  // actually dups in Buf_Comp
  const u4 x, y, z;
  const size_t len;
  const size_t anchor512_len;  // for spline

  // state
  bool is_comp;

  // arrays
  GPU_unique_dptr<E[]> d_ectrl;
  GPU_unique_dptr<T[]> d_anchor;
  GPU_unique_dptr<B[]> d_compressed;
  GPU_unique_hptr<B[]> h_compressed;
  GPU_unique_dptr<Freq[]> d_hist;
  GPU_unique_hptr<Freq[]> h_hist;
  GPU_unique_dptr<Freq[]> d_top1;
  GPU_unique_hptr<Freq[]> h_top1;

  std::unique_ptr<Buf_Outlier> buf_outlier;
  std::unique_ptr<Buf_HF> buf_hf;

  constexpr static size_t BLK = 8;  // for spline
  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

 private:
  static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

  static size_t set_anchor_len(u4 x, u4 y, u4 z)
  {
    return _div(x, BLK) * _div(y, BLK) * _div(z, BLK);
  }

 public:
  impl(u4 x, u4 y, u4 z, BufToggle_Comp* toggle) :
      x(x), y(y), z(z), len(x * y * z), anchor512_len(set_anchor_len(x, y, z))
  {
    if (toggle->use_quant) d_ectrl = MAKE_UNIQUE_DEVICE(E, len);
    if (toggle->use_outlier) buf_outlier = std::make_unique<Buf_Outlier>(len * OUTLIER_RATIO);
    if (toggle->use_anchor) d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
    if (toggle->use_hist) {
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
    }
    if (toggle->use_compressed) {
      d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
    }
    if (toggle->use_top1) {
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, 1);
    }
  }

  impl(u4 x, u4 y, u4 z, bool _is_comp) :
      is_comp(_is_comp), x(x), y(y), z(z), len(x * y * z), anchor512_len(set_anchor_len(x, y, z))
  {
    // align 4Ki for (essentially) FZG
    d_ectrl = MAKE_UNIQUE_DEVICE(E, ALIGN_4Ki(len));

    if (is_comp) {
      d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
      d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, 1);

      buf_outlier = std::make_unique<Buf_Outlier>(len / 5);
      buf_hf = std::make_unique<Buf_HF>(len, max_bklen);
    }
  }

  ~impl() {};

  void clear_buffer()
  {
    memset_device(d_ectrl.get(), len);
    memset_device(d_hist.get(), max_bklen);
    memset_device(d_anchor.get(), anchor512_len);
    memset_device(d_compressed.get(), len * 4 / 2);
    // TODO clear buf_outlier
  }
};

#define COMPBUF_IMPL(RET_TYPE)      \
  template <typename T, typename E> \
  RET_TYPE Buf_Comp<T, E>

namespace psz {

COMPBUF_IMPL()::Buf_Comp(u4 x, u4 y, u4 z, BufToggle_Comp* toggle) :
    x(x), y(y), z(z), len(x * y * z), pimpl(std::make_unique<impl>(x, y, z, toggle))
{
}

COMPBUF_IMPL()::Buf_Comp(u4 x, u4 y, u4 z, bool _is_comp) :
    is_comp(_is_comp),
    x(x),
    y(y),
    z(z),
    len(x * y * z),
    pimpl(std::make_unique<impl>(x, y, z, _is_comp))
{
}

COMPBUF_IMPL()::~Buf_Comp(){};

COMPBUF_IMPL(void)::clear_buffer() { pimpl->clear_buffer(); }

COMPBUF_IMPL(void)::clear_top1() { memset_device(pimpl->d_top1.get(), 1); }

// getters: array
COMPBUF_IMPL(E*)::ectrl_d() const { return pimpl->d_ectrl.get(); }
COMPBUF_IMPL(stdlen3)::ectrl_len3() const { return stdlen3{x, y, z}; }

COMPBUF_IMPL(Freq*)::hist_d() const { return pimpl->d_hist.get(); }
COMPBUF_IMPL(Freq*)::hist_h() const { return pimpl->h_hist.get(); }

COMPBUF_IMPL(Freq*)::top1_d() const { return pimpl->d_top1.get(); }
COMPBUF_IMPL(Freq*)::top1_h() const
{
  memcpy_allkinds<D2H>(pimpl->h_top1.get(), pimpl->d_top1.get(), 1);
  return pimpl->h_top1.get();
}

COMPBUF_IMPL(B*)::compressed_d() const { return pimpl->d_compressed.get(); }
COMPBUF_IMPL(B*)::compressed_h() const { return pimpl->h_compressed.get(); }

COMPBUF_IMPL(T*)::outlier_val_d() const { return pimpl->buf_outlier->val(); }
COMPBUF_IMPL(M*)::outlier_idx_d() const { return pimpl->buf_outlier->idx(); }
COMPBUF_IMPL(M)::outlier_num() const { return pimpl->buf_outlier->num_outliers(); }

COMPBUF_IMPL(T*)::anchor_d() const { return pimpl->d_anchor.get(); }
COMPBUF_IMPL(size_t)::anchor_len() const { return pimpl->anchor512_len; }
COMPBUF_IMPL(stdlen3)::anchor_len3() const
{
  auto _div = [](size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };
  return stdlen3{_div(x, BLK), _div(y, BLK), _div(z, BLK)};
}

template <typename T>
using Buf_Outlier = _portable::compact_gpu<T>;

template <typename E>
using Buf_HF = phf::Buf<E>;

COMPBUF_IMPL(Buf_Outlier<T>*)::buf_outlier() const { return pimpl->buf_outlier.get(); }

COMPBUF_IMPL(Buf_HF<E>*)::buf_hf() const { return pimpl->buf_hf.get(); }

}  // namespace psz

// instantiation
template class psz::Buf_Comp<f4, u1>;
template class psz::Buf_Comp<f4, u2>;
template class psz::Buf_Comp<f8, u1>;
template class psz::Buf_Comp<f8, u2>;
