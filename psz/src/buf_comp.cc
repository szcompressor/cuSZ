#include "mem/buf_comp.hh"

#include "cusz/type.h"
#include "kernel.hh"
#include "kernel/launch.inl"

// Dummy-kernel launch lives in buf_comp_dummy.cu (CUDA firewall).
namespace psz::buf_comp_dummy {
void launch();
}

namespace {

size_t set_top1_nblk(psz_len len)
{
  auto len3 = dim3(len.x, len.y, len.z);
  auto ndim = psz::config::utils::ndim(len3);

  auto flatten_grid = [](dim3 grid) { return static_cast<size_t>(grid.x) * grid.y * grid.z; };

  if (ndim == 1) return flatten_grid(psz::config::c_lorenzo<1>::thread_grid(dim3(len.x, 1, 1)));
  if (ndim == 2) return flatten_grid(psz::config::c_lorenzo<2, 32, 32>::thread_grid(len3));
  return flatten_grid(psz::config::c_lorenzo<3>::thread_grid(len3));
}

}  // namespace

template <typename T, typename E>
struct psz::Buf_Comp<T, E>::impl {
  const psz_len len;
  const size_t len_linear;
  const size_t len_linear_anchor;  // for spline
  const size_t len_top1;

  // state
  bool is_comp;

  // arrays
  GPU_unique_dptr<E[]> d_eq;
  GPU_unique_dptr<T[]> d_decode_fused;  // tile-order decode (2D HFR-family) scratch
  size_t eq_len_ = 0;                   // padded eq length (tile-order) or aligned linear
  GPU_unique_dptr<BYTE[]> d_compressed;
  GPU_unique_hptr<BYTE[]> h_compressed;
  GPU_unique_dptr<Freq[]> d_hist;
  GPU_unique_hptr<Freq[]> h_hist;
  GPU_unique_dptr<Freq[]> d_top1;
  GPU_unique_hptr<Freq[]> h_top1;

  std::unique_ptr<Buf_Outlier> buf_outlier;
  std::unique_ptr<Buf_Outlier2> buf_outlier2;
  std::unique_ptr<Buf_HF> buf_hf;
  std::unique_ptr<Buf_LC> buf_lc;

  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

  // spline-specific: declare
  GPU_unique_dptr<T[]> d_anchor;
  GPU_unique_dptr<T[]> d_pe;
  GPU_unique_hptr<T[]> h_pe;

  // spline variant selector (0 = y25/BLK16, 1 = y24/BLK8).
  int spline_variant = 0;
  int anchor_blk() const { return spline_variant == 1 ? BLK8 : BLK16; }

 private:
  static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

  // 1Ki: lrz2d: 32x32
  // 2Ki: lrz3d/spl-y24: 32x8x8
  // 4Ki: spl-y25: 16x16x16 or 64x64 (four 16x8x8 chunks).
  // 1D cases are trivially linear.
  static size_t set_eq_padded(psz_len l, bool y25 = false)
  {
    size_t linear = (size_t)l.x * l.y * l.z;
    size_t aligned = ALIGN_4Ki(linear);
    if (y25) {  // 4Ki
      size_t padded = (l.z > 1) ? _div(l.x, 16) * _div(l.y, 16) * _div(l.z, 16) * 4096
                                : _div(l.x, 64) * _div(l.y, 64) * 4096;
      return aligned > padded ? aligned : padded;
    }
    if (l.z > 1) {  // 3D: 32x8x8, 2Ki
      size_t padded3d = _div(l.x, 32) * _div(l.y, 8) * _div(l.z, 8) * 2048;
      return aligned > padded3d ? aligned : padded3d;
    }
    if (l.y > 1) {  // 2D: 32x32, 1Ki
      size_t padded2d = _div(l.x, 32) * _div(l.y, 32) * 1024;
      return aligned > padded2d ? aligned : padded2d;
    }
    return aligned;
  }

  // element count to append past eq_len for per-tile outlier-cell staging (blk_cells_tail).
  static size_t set_outlier_tail_elems(psz_len l, bool y25, size_t eq_len)
  {
    size_t chunk = y25 ? 4096 : (l.z > 1 ? 2048 : 1024);  // 1D/2D share the 1Ki chunk
    size_t magnitude = y25 ? 12 : (l.z > 1 ? 11 : 10);
    size_t cap = magnitude == 12   ? psz::HFR_PBK_C12::MaxNumUnpred
                 : magnitude == 11 ? psz::HFR_PBK_C11::MaxNumUnpred
                                   : psz::HFR_PBK_C10::MaxNumUnpred;
    size_t n_tiles = eq_len / chunk;
    size_t tail_bytes = n_tiles * cap * sizeof(psz::OutlierCell);
    return (tail_bytes + sizeof(E) - 1) / sizeof(E);
  }

  // y24 cap for both y24 and y25
  static size_t set_anchor_len(u4 x, u4 y, u4 z)
  { return _div(x, BLK8) * _div(y, BLK8) * _div(z, BLK8); }

  static size_t set_anchor_len(psz_len len)
  { return _div(len.x, BLK8) * _div(len.y, BLK8) * _div(len.z, BLK8); }

 public:
  impl(psz_len _len, BufToggle_Comp* toggle) :
      len(_len),
      len_linear(_len.x * _len.y * _len.z),
      len_linear_anchor(set_anchor_len(_len)),
      len_top1(set_top1_nblk(_len))
  {
    const auto outlier_cap = static_cast<size_t>(len_linear * OUTLIER_RATIO);
    const auto spfmt_max_bytes =
        std::max(sizeof(T) + sizeof(u4), sizeof(_ptb::compact_cell<T, M>)) * outlier_cap;
    const auto bitr_input_max_bytes = len_linear_anchor * sizeof(T) + spfmt_max_bytes;
    const auto codec_max_bytes = len_linear * sizeof(E);
    const auto rtr_input_max_bytes = codec_max_bytes + bitr_input_max_bytes;
    if (toggle->use_lc)
      buf_lc = std::make_unique<Buf_LC>(
          len_linear * sizeof(E), bitr_input_max_bytes, rtr_input_max_bytes, rtr_input_max_bytes);
    // spline profiles interpolation direction whatever the predictor, and this is
    // ERR_HISTO_LEN elements: too small to gate
    d_pe = MAKE_UNIQUE_DEVICE(T, ERR_HISTO_LEN);
    h_pe = MAKE_UNIQUE_HOST(T, ERR_HISTO_LEN);

    if (toggle->use_quant) d_eq = MAKE_UNIQUE_DEVICE(E, len_linear);
    if (toggle->use_outlier) {
      buf_outlier = std::make_unique<Buf_Outlier>(len_linear * OUTLIER_RATIO);
      buf_outlier2 = std::make_unique<Buf_Outlier2>(len_linear * OUTLIER_RATIO);
    }
    if (toggle->use_anchor) {
      d_anchor = MAKE_UNIQUE_DEVICE(T, len_linear_anchor);
      // the spline encoder writes only the anchor slots its own grid covers; an
      // unzeroed slot outside it makes the first reconstruction differ from later ones
      memset_device(d_anchor.get(), len_linear_anchor);
    }
    if (toggle->use_hist) {
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
    }
    if (toggle->use_compressed) {
      d_compressed = MAKE_UNIQUE_DEVICE(BYTE, len_linear * sizeof(E) * 3 / 2);
      h_compressed = MAKE_UNIQUE_HOST(BYTE, len_linear * sizeof(E) * 3 / 2);
    }
    if (toggle->use_top1) {
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, len_top1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, len_top1);
    }
  }

  impl(
      psz_len _len, bool _is_comp, bool use_HFR = false, bool alloc_eq = true,
      bool use_sublen_1ki = false, bool tile_order = false, bool y25_tile = false) :
      is_comp(_is_comp),
      len(_len),
      len_linear(_len.x * _len.y * _len.z),
      len_linear_anchor(set_anchor_len(_len)),
      len_top1(set_top1_nblk(_len))
  {
    // 4Ki, maximum according to spl-y25
    // 2D/3D kernels write eq tiled even when tile_order is false: always size for tiled.
    size_t const eq_len = set_eq_padded(_len, y25_tile);
    eq_len_ = eq_len;
    // FIXME: compat mode FZG
    // spl-y25 requires d_eq for decompression due to per-level clustering.
    // lrz and spl-y24 decode eq directly to output buffer
    // 1D is not tile-ordered but its HFR-family cells also ride the eq tail (blk_cells_tail).
    size_t const outlier_tail =
        (tile_order or use_HFR) ? set_outlier_tail_elems(_len, y25_tile, eq_len) : 0;
    if (is_comp or alloc_eq) d_eq = MAKE_UNIQUE_DEVICE(E, eq_len + outlier_tail);
    // HF decodes eq + scattered outliers into tiles.
    if (not is_comp and tile_order) d_decode_fused = MAKE_UNIQUE_DEVICE(T, eq_len);
    // HF encodes/decodes every 1Ki/2Ki/4Ki.
    size_t hf_len = tile_order ? eq_len : len_linear;
    buf_hf = std::make_unique<Buf_HF>(hf_len, max_bklen, -1, use_HFR, false, use_sublen_1ki);
    const auto outlier_cap = static_cast<size_t>(len_linear * OUTLIER_RATIO);
    const auto spfmt_max_bytes =
        std::max(sizeof(T) + sizeof(u4), sizeof(_ptb::compact_cell<T, M>)) * outlier_cap;
    const auto bitr_input_max_bytes = len_linear_anchor * sizeof(T) + spfmt_max_bytes;
    const auto codec_max_bytes = len_linear * sizeof(E);
    const auto rtr_input_max_bytes = codec_max_bytes + bitr_input_max_bytes;
    buf_lc = std::make_unique<Buf_LC>(
        len_linear * sizeof(E), bitr_input_max_bytes, rtr_input_max_bytes, rtr_input_max_bytes);

    if (is_comp) {
      d_anchor = MAKE_UNIQUE_DEVICE(T, len_linear_anchor);
      d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
      h_hist = MAKE_UNIQUE_HOST(Freq, max_bklen);
      d_compressed = MAKE_UNIQUE_DEVICE(BYTE, len_linear * sizeof(E) * 3 / 2);
      h_compressed = MAKE_UNIQUE_HOST(BYTE, len_linear * sizeof(E) * 3 / 2);
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, len_top1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, len_top1);

      buf_outlier = std::make_unique<Buf_Outlier>(len_linear * OUTLIER_RATIO);
      buf_outlier2 = std::make_unique<Buf_Outlier2>(len_linear * OUTLIER_RATIO);

      d_pe = MAKE_UNIQUE_DEVICE(T, ERR_HISTO_LEN);
      h_pe = MAKE_UNIQUE_HOST(T, ERR_HISTO_LEN);

      // call dummy during expensive init
      psz::buf_comp_dummy::launch();
    }

    if (d_eq) memset_device(d_eq.get(), eq_len);
  }

  ~impl() {};

  void clear_buffer()
  {
    memset_device(d_eq.get(), len_linear);
    memset_device(d_hist.get(), max_bklen);
    memset_device(d_anchor.get(), len_linear_anchor);
    memset_device(d_compressed.get(), len_linear * sizeof(E) * 3 / 2);
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

COMPBUF_IMPL()::Buf_Comp(
    psz_len _len, bool _is_comp, bool use_HFR, bool alloc_eq, bool use_sublen_1ki, bool tile_order,
    bool y25_tile) :
    is_comp(_is_comp),
    len(_len),
    len_linear(_len.x * _len.y * _len.z),
    pimpl(
        std::make_unique<impl>(
            _len, _is_comp, use_HFR, alloc_eq, use_sublen_1ki, tile_order, y25_tile))
{
}

COMPBUF_IMPL()::~Buf_Comp(){};

COMPBUF_IMPL(void)::clear_buffer() { pimpl->clear_buffer(); }

COMPBUF_IMPL(void)::clear_top1() { memset_device(pimpl->d_top1.get(), pimpl->len_top1); }

// getters: array
COMPBUF_IMPL(E*)::eq_d() const { return pimpl->d_eq.get(); }
COMPBUF_IMPL(psz_len)::eq_len3() const { return len; }
COMPBUF_IMPL(T*)::decode_fused_d() const { return pimpl->d_decode_fused.get(); }
COMPBUF_IMPL(size_t)::eq_len() const { return pimpl->eq_len_; }
COMPBUF_IMPL(void)::alloc_decode_fused()  // FIXME: bin_pred reconstructs on a compress-side buf.
{
  if (not pimpl->d_decode_fused) pimpl->d_decode_fused = MAKE_UNIQUE_DEVICE(T, pimpl->eq_len_);
}
using psz::OutlierCell;
COMPBUF_IMPL(OutlierCell*)::block_outliers_d() const
{
  if (not pimpl->d_eq) return nullptr;
  return (OutlierCell*)(pimpl->d_eq.get() + pimpl->eq_len_);
}

COMPBUF_IMPL(Freq*)::hist_d() const { return pimpl->d_hist.get(); }
COMPBUF_IMPL(Freq*)::hist_h() const { return pimpl->h_hist.get(); }

COMPBUF_IMPL(Freq*)::top1_d() const { return pimpl->d_top1.get(); }
COMPBUF_IMPL(Freq*)::top1_h() const
{
  memcpy_allkinds<D2H>(pimpl->h_top1.get(), pimpl->d_top1.get(), pimpl->len_top1);
  return pimpl->h_top1.get();
}

COMPBUF_IMPL(size_t)::top1_nblk() const { return pimpl->len_top1; }

COMPBUF_IMPL(BYTE*)::compressed_d() const { return pimpl->d_compressed.get(); }
COMPBUF_IMPL(BYTE*)::compressed_h() const { return pimpl->h_compressed.get(); }

COMPBUF_IMPL(void*)::outlier2_validx_d() const { return pimpl->buf_outlier2->val_idx_d(); }
COMPBUF_IMPL(M)::outlier2_host_get_num() const { return pimpl->buf_outlier2->host_get_num(); }
COMPBUF_IMPL(size_t)::outlier2_max_allowed_num() const
{ return pimpl->buf_outlier2->max_allowed_num(); }

COMPBUF_IMPL(T*)::anchor_d() const { return pimpl->d_anchor.get(); }
COMPBUF_IMPL(size_t)::anchor_len() const
{
  auto a = anchor_len3();
  return (size_t)a.x * a.y * a.z;
}
COMPBUF_IMPL(psz_len)::anchor_len3() const
{
  auto _div = [](size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };
  auto blk = pimpl->anchor_blk();
  return {_div(len.x, blk), _div(len.y, blk), _div(len.z, blk)};
}

COMPBUF_IMPL(void)::set_spline_variant(int v) { pimpl->spline_variant = v; }

COMPBUF_IMPL(T*)::profiled_errors_d() const { return pimpl->d_pe.get(); };
COMPBUF_IMPL(T*)::profiled_errors_h() const { return pimpl->h_pe.get(); };
COMPBUF_IMPL(M)::profiled_errors_len() const { return ERR_HISTO_LEN; };

template <typename T>
using Buf_Outlier = _ptb::compact_gpu<T>;

template <typename T>
using Buf_Outlier2 = _ptb::compact_GPU_DRAM2<T, M>;

template <typename E>
using Buf_HF = phf::Buf<E>;
using Buf_LC = LC_Buf;

COMPBUF_IMPL(Buf_Outlier2<T>*)::buf_outlier2() const { return pimpl->buf_outlier2.get(); }

COMPBUF_IMPL(Buf_HF<E>*)::buf_hf() const { return pimpl->buf_hf.get(); }
COMPBUF_IMPL(Buf_LC*)::buf_lc() const { return pimpl->buf_lc.get(); }

}  // namespace psz

// instantiation
template class psz::Buf_Comp<f4, u1>;
template class psz::Buf_Comp<f4, u2>;
template class psz::Buf_Comp<f4, u4>;
template class psz::Buf_Comp<f8, u1>;
template class psz::Buf_Comp<f8, u2>;
template class psz::Buf_Comp<f8, u4>;
