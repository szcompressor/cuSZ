#ifndef PSZ_COMPBUF_HH
#define PSZ_COMPBUF_HH

#include <array>

#include "cusz/context.h"
#include "cusz/type.h"
#include "hf_hl.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"

// segment: ANCHOR is contiguous SPFMT.
#define HEADER 0
#define ENCODED 1
#define ANCHOR 2
#define SPFMT 3
#define END 4
#define ENC_PASS1_END 4
#define ENC_PASS2_END 5

using stdlen3 = std::array<size_t, 3>;

namespace psz {

struct CompressorBufferToggle {
  bool err_ctrl_quant;
  bool compact_outlier;
  bool anchor;
  bool profile_error;
  bool histogram;
  bool compressed;
  bool top1;
};

template <typename DType, typename EType = u2>
class CompressorBuffer {
 public:
  using T = DType;
  using E = EType;
  using FP = T;
  using M = u4;
  using Freq = u4;
  using B = u1;
  using BYTE = u1;
  using H = u4;
  using H4 = u4;
  using H8 = u8;
  using Compact = _portable::compact_gpu<T>;
  using hf_mem_t = phf::Buf<E>;

  GPU_unique_dptr<E[]> d_ectrl;
  GPU_unique_dptr<B[]> d_compressed;
  GPU_unique_hptr<B[]> h_compressed;
  GPU_unique_dptr<Freq[]> d_hist;
  GPU_unique_hptr<Freq[]> h_hist;
  GPU_unique_dptr<Freq[]> d_top1;
  GPU_unique_hptr<Freq[]> h_top1;

  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

  // spline-specific: declare
  constexpr static int BLK = 16;
  constexpr static int ERR_HISTO_LEN = 36;
  GPU_unique_dptr<T[]> d_anchor;
  GPU_unique_dptr<T[]> d_pe;
  GPU_unique_hptr<T[]> h_pe;

  Compact* compact;
  bool is_comp;
  const u4 x, y, z;
  const size_t len;
  const size_t anchor512_len;  // for spline

  // encapsulations
  int hist_generic_grid_dim;
  int hist_generic_block_dim;
  int hist_generic_shmem_use;
  int hist_generic_repeat;
  BYTE* comp_codec_out{nullptr};
  size_t comp_codec_outlen{0};
  uint32_t nbyte[ENC_PASS1_END];
  [[deprecated]] float time_sp;
  double eb, eb_r, ebx2, ebx2_r;

  psz_header* header_ref;

  hf_mem_t* buf_hf;

 private:
  static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

  static size_t set_len_anchor_512(u4 x, u4 y, u4 z)
  {
    return _div(x, BLK) * _div(y, BLK) * _div(z, BLK);
  }

 public:
  CompressorBuffer(u4 x, u4 y, u4 z, CompressorBufferToggle* toggle) :
      x(x), y(y), z(z), len(x * y * z), anchor512_len(set_len_anchor_512(x, y, z))
  {
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

  CompressorBuffer(u4 x, u4 y = 1, u4 z = 1, bool _is_comp = true) :
      is_comp(_is_comp),
      x(x),
      y(y),
      z(z),
      len(x * y * z),
      anchor512_len(set_len_anchor_512(x, y, z))
  {
    // align 4Ki for (essentially) FZG
    d_ectrl = MAKE_UNIQUE_DEVICE(E, ALIGN_4Ki(len));

    if (is_comp) {
      compact = new Compact(len / 5);
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
      d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, 1);

      buf_hf = new phf::Buf<E>(len, max_bklen);

      // spline-specific: allocate
      d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
      d_pe = MAKE_UNIQUE_DEVICE(T, ERR_HISTO_LEN);
      h_pe = MAKE_UNIQUE_HOST(T, ERR_HISTO_LEN);
    }
  }

  // void register_header(psz_context* ctx) { header_ref = ctx->header; }
  void register_header(psz_header* header) { header_ref = header; }

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

  B* compressed() const { return d_compressed.get(); }
  B* compressed_h() const { return d_compressed.get(); }

  T* compact_val() const { return compact->val(); }
  M* compact_idx() const { return compact->idx(); }
  M compact_num_outliers() const { return compact->num_outliers(); }
  Compact* outlier() { return compact; }

  // spline-specific: getter
  T* anchor() const { return d_anchor.get(); }
  size_t anchor_len() const { return anchor512_len; }
  stdlen3 anchor_len3() const { return stdlen3{_div(x, BLK), _div(y, BLK), _div(z, BLK)}; }
  T* profiled_errors() const { return d_pe.get(); };
  T* profiled_errors_h() const { return h_pe.get(); };
  M profiled_errors_len() const { return ERR_HISTO_LEN; };
};

}  // namespace psz

#endif /* PSZ_COMPBUF_HH */
