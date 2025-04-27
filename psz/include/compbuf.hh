#include <array>

#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"

using stdlen3 = std::array<size_t, 3>;

namespace psz {

struct CompressorBufferToggle {
  bool err_ctrl_quant;
  bool compact_outlier;
  bool anchor;
  bool histogram;
  bool compressed;
  bool top1;
};

template <typename DType>
class CompressorBuffer {
 public:
  using T = DType;
  using E = u2;
  using FP = T;
  using M = u4;
  using Freq = u4;
  using B = u1;
  using Compact = _portable::compact_gpu<T>;

  GPU_unique_dptr<E[]> d_ectrl;
  GPU_unique_dptr<T[]> d_anchor;
  GPU_unique_dptr<B[]> d_compressed;
  GPU_unique_hptr<B[]> h_compressed;
  GPU_unique_dptr<Freq[]> d_hist;
  GPU_unique_hptr<Freq[]> h_hist;
  GPU_unique_dptr<Freq[]> d_top1;
  GPU_unique_hptr<Freq[]> h_top1;

  Compact* compact;
  bool const is_comp;

  constexpr static size_t BLK = 8;  // for spline
  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

  u4 const x, y, z;
  size_t const len;
  size_t const anchor512_len;  // for spline

 private:
  static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

  static size_t set_len_anchor_512(u4 x, u4 y, u4 z)
  {
    return _div(x, BLK) * _div(y, BLK) * _div(z, BLK);
  }

 public:
  CompressorBuffer(
      u4 x, u4 y = 1, u4 z = 1, bool _is_comp = true, CompressorBufferToggle* toggle = nullptr) :
      is_comp(_is_comp),
      x(x),
      y(y),
      z(z),
      len(x * y * z),
      anchor512_len(set_len_anchor_512(x, y, z))
  {
    if (not toggle) {
      // align 4Ki for (essentially) FZG
      d_ectrl = MAKE_UNIQUE_DEVICE(E, ALIGN_4Ki(len));

      if (is_comp) {
        compact = new Compact(len / 5);
        d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
        d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
        h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
        d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
        h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
        d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
        h_top1 = MAKE_UNIQUE_HOST(Freq, 1);
      }
    }
    else {
      if (toggle->err_ctrl_quant) d_ectrl = MAKE_UNIQUE_DEVICE(E, len);
      if (toggle->compact_outlier) compact = new Compact(len / 5);
      if (toggle->anchor) d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
      if (toggle->histogram) {
        d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
        h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
      }
      if (toggle->compressed) {
        d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
        h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
      }
      if (toggle->top1) {
        d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
        h_top1 = MAKE_UNIQUE_HOST(Freq, 1);
      }
    }
  }

  ~CompressorBuffer()
  {
    if (is_comp) delete compact;
  }

  // utils
  CompressorBuffer* clear_buffer()
  {
    memset_device(d_ectrl.get(), len);  // TODO FZG padding
    memset_device(d_hist.get(), max_bklen);
    memset_device(d_anchor.get(), anchor512_len);
    memset_device(d_compressed.get(), len * 4 / 2);
    // TODO clear compact
    return this;
  }
  // getter
  E* ectrl() const { return d_ectrl.get(); }

  Freq* hist() const { return d_hist.get(); }
  Freq* top1() const { return d_top1.get(); }
  Freq* top1_h() const
  {
    memcpy_allkinds<D2H>(h_top1.get(), d_top1.get(), 1);
    return h_top1.get();
  }
  // For iterative run, it is useful to clear up.
  void clear_top1() { memset_device(d_top1.get(), 1); }

  stdlen3 ectrl_len3() const { return stdlen3{x, y, z}; }

  T* anchor() const { return d_anchor.get(); }
  size_t anchor_len() const { return anchor512_len; }
  stdlen3 anchor_len3() const { return stdlen3{_div(x, BLK), _div(y, BLK), _div(z, BLK)}; }

  B* compressed() const { return d_compressed.get(); }
  B* compressed_h() const { return d_compressed.get(); }

  T* compact_val() const { return compact->val(); }
  M* compact_idx() const { return compact->idx(); }
  M compact_num_outliers() const { return compact->num_outliers(); }
  Compact* outlier() { return compact; }
};

}  // namespace psz
