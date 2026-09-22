#include "mem/buf_comp.hh"

#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "cusz/header.h"
#include "cusz/type.h"
#include "kernel.hh"
#include "kernel/launch.inl"
#include "mem/pool.h"
#include "module.hh"

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

size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

int ndim(psz_len l) { return psz::config::utils::ndim(dim3(l.x, l.y, l.z)); }

// 1Ki: lrz2d: 32x32
// 2Ki: lrz3d/spl-y24: 32x8x8
// 4Ki: spl-y25: 16x16x16 or 64x64 (four 16x8x8 chunks).
// 1D cases are trivially linear.
size_t set_eq_padded(psz_len l, bool y25 = false)
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

// 4Ki at max (spl-y25)
size_t hf_len_of(psz_len l)
{
  return ndim(l) >= 2 ? std::max(set_eq_padded(l), set_eq_padded(l, true))
                      : (size_t)l.x * l.y * l.z;
}

template <typename E>
size_t set_outlier_tail_elems(psz_len l, bool y25, size_t eq_len)
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

size_t eq_bytes(psz_len l, bool y25, size_t eq, bool eq4)
{
  bool const tile_nd = ndim(l) >= 2;
  size_t const u2_bytes = (eq + (tile_nd ? set_outlier_tail_elems<u2>(l, y25, eq) : 0)) * sizeof(u2);
  size_t const u4_bytes = eq4 ? (eq + set_outlier_tail_elems<u4>(l, y25, eq)) * sizeof(u4) : 0;
  return std::max(u2_bytes, u4_bytes);
}

}  // namespace

struct stage_buf : _ptb::buf_base {
  using _ptb::buf_base::buf_base;
  GPU_unique_dptr<u1[]> d_data, d_state;

  void init() override
  {
    d_data = MAKE_UNIQUE_DEVICE(u1, data().bytes());
    d_state = MAKE_UNIQUE_DEVICE(u1, state().bytes());
    attach(d_data.get(), d_state.get());
  }
  void reset(void*) override {}  // Buf_Comp::reset drives every child
};

template <typename T>
struct lrz_buf : stage_buf {
  enum : int { EQ, DECODE_FUSED, TOP1 };
  size_t const eq_len;
  size_t const top1_len;
  GPU_unique_hptr<u4[]> h_top1;

  lrz_buf(psz_len l, bool is_comp, bool eq4) :
      stage_buf(plan(l, is_comp, eq4)), eq_len(set_eq_padded(l)), top1_len(set_top1_nblk(l))
  {
    if (is_comp) h_top1 = MAKE_UNIQUE_HOST(u4, top1_len);
  }

  static _ptb::mem_plan::tables plan(psz_len l, bool is_comp, bool eq4)
  {
    bool const tile_nd = ndim(l) >= 2;
    auto const eq = set_eq_padded(l);
    return {
        // arrays
        {{EQ, U1, is_comp ? eq_bytes(l, false, eq, eq4) : 0},
         {DECODE_FUSED, _ptb::dtype_of<T>(), (not is_comp and tile_nd) ? eq : 0}},
        // metadata
        {{TOP1, U4, is_comp ? set_top1_nblk(l) : 0}}};
  }

  template <typename E>
  E* eq() const
  { return (E*)data().ptr_of<u1>(EQ); }
  T* decode_fused() const { return data().ptr_of<T>(DECODE_FUSED); }
  u4* top1() const { return state().ptr_of<u4>(TOP1); }
};

template <typename T>
struct spl_buf : stage_buf {
  constexpr static int BLK16 = 16;  // y25 (2D+3D)
  constexpr static int BLK8 = 8;    // y24 (lean 3D)
  constexpr static int ERR_HISTO_LEN = 36;

  enum : int { EQ, ANCHOR, DECODE_FUSED, PE };
  psz_len const len;
  size_t const eq_len_y24, eq_len_y25;
  psz_predictor variant = SplineY25;
  GPU_unique_hptr<T[]> h_pe;

  spl_buf(psz_len l, bool is_comp, bool eq4) :
      stage_buf(plan(l, is_comp, eq4)),
      len(l),
      eq_len_y24(set_eq_padded(l)),
      eq_len_y25(set_eq_padded(l, ndim(l) >= 2))
  {
    if (is_comp) h_pe = MAKE_UNIQUE_HOST(T, ERR_HISTO_LEN);
  }

  // y24 cap for both y24 and y25
  static size_t anchor_cap(psz_len l)
  { return _div(l.x, BLK8) * _div(l.y, BLK8) * _div(l.z, BLK8); }

  static _ptb::mem_plan::tables plan(psz_len l, bool is_comp, bool eq4)
  {
    bool const tile_nd = ndim(l) >= 2;
    auto const eq24 = set_eq_padded(l), eq25 = set_eq_padded(l, tile_nd);
    // spl-y25: use d_eq for decomp per-level
    // lrz and spl-y24 decode eq directly to output buffer
    auto const eq_frame =
        is_comp ? std::max(eq_bytes(l, false, eq24, eq4), eq_bytes(l, tile_nd, eq25, eq4))
                : eq_bytes(l, tile_nd, eq25, eq4);
    return {
        // arrays
        {{EQ, U1, eq_frame},
         {ANCHOR, _ptb::dtype_of<T>(), is_comp ? anchor_cap(l) : 0},
         {DECODE_FUSED, _ptb::dtype_of<T>(),
          (not is_comp and tile_nd) ? std::max(eq24, eq25) : 0}},
        // metadata
        {{PE, _ptb::dtype_of<T>(), is_comp ? (size_t)ERR_HISTO_LEN : 0}}};
  }

  size_t eq_len() const { return variant == SplineY24 ? eq_len_y24 : eq_len_y25; }
  psz_len anchor_len3() const
  {
    size_t const blk = variant == SplineY24 ? BLK8 : BLK16;
    return {_div(len.x, blk), _div(len.y, blk), _div(len.z, blk)};
  }

  template <typename E>
  E* eq() const
  { return (E*)data().ptr_of<u1>(EQ); }
  T* anchor() const { return data().ptr_of<T>(ANCHOR); }
  T* decode_fused() const { return data().ptr_of<T>(DECODE_FUSED); }
  T* pe() const { return state().ptr_of<T>(PE); }
};

template <typename T>
struct psz::Buf_Comp<T>::impl {
  const psz_len len;
  const size_t len_linear;
  const size_t len_top1;

  // state
  bool is_comp;
  int const nstage;
  bool const eq4;
  size_t const archive_capacity;

  _ptb::pool pool_predict, pool_encode1, pool_encode2, pool_archive;
  _ptb::mem_plan::total bound_predict{0, 0}, bound_encode1{0, 0}, bound_encode2{0, 0};
  bool bound = false;
  psz_ppl ppl{};
  psz_predictor predictor = SplineY25;

  // arrays
  GPU_unique_dptr<T[]> d_decode_fused;
  BYTE* wired_archive = nullptr;
  BYTE* archive() const { return wired_archive ? wired_archive : (BYTE*)pool_archive.data(); }
  GPU_unique_hptr<BYTE[]> h_compressed;

  std::unique_ptr<lrz_buf<T>> buf_lrz;
  std::unique_ptr<spl_buf<T>> buf_spl;
  std::unique_ptr<Buf_Outlier2> buf_outlier2;
  std::unique_ptr<Buf_HF> buf_hf;
  std::unique_ptr<Buf_HFR> buf_hfr;
  std::unique_ptr<Buf_FZG> buf_fzg;
  std::unique_ptr<Buf_LC> buf_lc1, buf_lc2;

  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

  template <typename B>
  static _ptb::mem_plan::tree leaf_of(B* b)
  {
    if (not b) return _ptb::mem_plan::tree(_ptb::mem_plan::total{0, 0});
    if constexpr (std::is_base_of_v<_ptb::buf_base, B>)
      return _ptb::leaf(*b);
    else
      return _ptb::mem_plan::tree(
          _ptb::mem_plan::total{b->planned_data_bytes(), b->planned_state_bytes()},
          [b](void* d, void* s) { b->attach(d, s); });
  }

  _ptb::mem_plan::tree stage_predict() const
  { return (leaf_of(buf_lrz.get()) | leaf_of(buf_spl.get())) + leaf_of(buf_outlier2.get()); }
  _ptb::mem_plan::tree stage_encode1(bool ppl_eq4) const
  {
    if (ppl_eq4) return leaf_of(buf_hfr.get());
    return leaf_of(buf_hf.get()) + (leaf_of(buf_fzg.get()) | leaf_of(buf_lc1.get()));
  }
  _ptb::mem_plan::tree stage_encode2() const { return leaf_of(buf_lc2.get()); }

  bool spline() const { return psz::_2609::is_spline(predictor); }
  template <typename E>
  E* eq() const
  { return spline() ? buf_spl->template eq<E>() : buf_lrz->template eq<E>(); }
  T* anchor() const { return buf_spl->anchor(); }
  T* decode_fused() const { return spline() ? buf_spl->decode_fused() : buf_lrz->decode_fused(); }
  Freq* top1() const { return buf_lrz->top1(); }
  T* pe() const { return buf_spl->pe(); }
  size_t eq_len() const { return spline() ? buf_spl->eq_len() : buf_lrz->eq_len; }

  void set_predictor(psz_predictor p)
  {
    predictor = p;
    if (psz::_2609::is_spline(p)) buf_spl->variant = p;
  }

  bool select(psz_ppl const& p)
  {
    if (not psz::_2609::valid(p)) return false;
    bool const ppl_eq4 = psz::_2609::needs_eq4(p);
    if (ppl_eq4 and not eq4) return false;
    if (p.codec1 != CodecNull and nstage < 2) return false;
    if (p.codec2 != CodecNull and nstage < 3) return false;
    set_predictor(p.predictor);
    bool const same =
        ppl.predictor == p.predictor and ppl.codec1 == p.codec1 and ppl.codec2 == p.codec2;
    if (bound and same) return true;
    if (bound) {
      if (bound_predict.state) memset_device((u1*)pool_predict.state(), bound_predict.state);
      if (bound_encode1.state) memset_device((u1*)pool_encode1.state(), bound_encode1.state);
      if (bound_encode2.state) memset_device((u1*)pool_encode2.state(), bound_encode2.state);
    }
    stage_predict().assign({0, 0}, pool_predict.data(), pool_predict.state());
    stage_encode1(ppl_eq4).assign({0, 0}, pool_encode1.data(), pool_encode1.state());
    stage_encode2().assign({0, 0}, pool_encode2.data(), pool_encode2.state());
    bound = true;
    ppl = p;
    return true;
  }

  impl(psz_len _len, bool _is_comp, BYTE* external_archive, int _nstage, bool _eq4) :
      is_comp(_is_comp),
      nstage(_nstage),
      eq4(_eq4 and _nstage >= 2),
      archive_capacity(_is_comp ? Buf_Comp<T>::compressed_max_bytes(_len, _nstage, eq4) : 0),
      len(_len),
      len_linear(_len.x * _len.y * _len.z),
      len_top1(set_top1_nblk(_len))
  {
    size_t const hf_len = hf_len_of(len);
    const auto outlier_cap = static_cast<size_t>(len_linear * OUTLIER_RATIO);
    const auto spfmt_max_bytes =
        std::max(sizeof(T) + sizeof(u4), sizeof(_ptb::compact_cell<T, M>)) * outlier_cap;
    const auto bitr_input_max_bytes = spl_buf<T>::anchor_cap(len) * sizeof(T) + spfmt_max_bytes;
    const auto codec_max_bytes = hf_len * sizeof(u2);  // use (padded) hf_len for TCMS
    const auto rtr_input_max_bytes = codec_max_bytes + bitr_input_max_bytes;
    const auto rtr_input_max_bytes_eq4 =
        hf_len * (eq4 ? sizeof(u4) : sizeof(u2)) + bitr_input_max_bytes;

    void* hf_archive_dst = nullptr;
    size_t encoded_in = 0, chunked_in_max = 0, decoded_max = rtr_input_max_bytes,
           decoded_max_eq4 = rtr_input_max_bytes_eq4;
    if (is_comp) {
      wired_archive = external_archive;
      if (nstage >= 2) {
        if (not wired_archive) pool_archive.allocate({archive_capacity, 0});
        hf_archive_dst = archive() + psz::_2609::pad8(sizeof(psz_header));
        h_compressed = MAKE_UNIQUE_HOST(BYTE, archive_capacity);
      }
      buf_outlier2 = std::make_unique<Buf_Outlier2>(outlier_cap, false, false);
      encoded_in = rtr_input_max_bytes_eq4;
      chunked_in_max = codec_max_bytes;
      decoded_max = 0;
      decoded_max_eq4 = 0;
      psz::buf_comp_dummy::launch();
    }

    buf_lrz = std::make_unique<lrz_buf<T>>(len, is_comp, eq4);
    buf_spl = std::make_unique<spl_buf<T>>(len, is_comp, eq4);
    if (nstage >= 2) {
      buf_hf = std::make_unique<Buf_HF>(hf_len, max_bklen, false, is_comp);
      if (eq4)
        buf_hfr = std::make_unique<Buf_HFR>(hf_len, max_bklen, false, is_comp, hf_archive_dst);
      buf_fzg = std::make_unique<Buf_FZG>(set_eq_padded(len), is_comp);
      buf_lc1 = std::make_unique<Buf_LC>(LC_TCMS, 0, chunked_in_max, decoded_max);
    }
    if (nstage >= 3)
      buf_lc2 = std::make_unique<Buf_LC>(LC_RTR, encoded_in, encoded_in, decoded_max_eq4);

    bound_predict = stage_predict().bytes();
    bound_encode1 = (stage_encode1(false) | stage_encode1(true)).bytes();
    bound_encode2 = stage_encode2().bytes();
    pool_predict.allocate(bound_predict);
    pool_encode1.allocate(bound_encode1);
    pool_encode2.allocate(bound_encode2);

    if (std::getenv("PSZ_DBG")) {
      auto const mib = [](size_t b) { return b / 1048576.0; };
      auto const grand = bound_predict + bound_encode1 + bound_encode2;

      printf(
          "  stage1 predict+outlier  data %8.2f MiB  state %8.3f MiB\n", mib(bound_predict.data),
          mib(bound_predict.state));
      printf(
          "  stage2 codec slot 1     data %8.2f MiB  state %8.3f MiB\n", mib(bound_encode1.data),
          mib(bound_encode1.state));
      printf(
          "  stage3 codec slot 2     data %8.2f MiB  state %8.3f MiB\n", mib(bound_encode2.data),
          mib(bound_encode2.state));
      printf(
          "  grand  1+2+3            data %8.2f MiB  state %8.3f MiB  archive %8.2f MiB\n",
          mib(grand.data), mib(grand.state),
          mib(is_comp and nstage >= 2 and not wired_archive ? archive_capacity : 0));
    }
  }

  ~impl() {};
};

#define COMPBUF_IMPL(RET_TYPE) \
  template <typename T>        \
  RET_TYPE Buf_Comp<T>

namespace psz {

COMPBUF_IMPL()::Buf_Comp(
    psz_len _len, bool _is_comp, BYTE* external_archive, int nstage, bool eq4) :
    is_comp(_is_comp),
    len(_len),
    len_linear(_len.x * _len.y * _len.z),
    pimpl(std::make_unique<impl>(_len, _is_comp, external_archive, nstage, eq4))
{
}

COMPBUF_IMPL(bool)::select(psz_ppl ppl) { return pimpl->select(ppl); }

COMPBUF_IMPL()::~Buf_Comp(){};

COMPBUF_IMPL(void)::reset(void* stream)
{
  if (pimpl->top1()) memset_device_async(pimpl->top1(), pimpl->len_top1, 0, stream);
  if (pimpl->pe()) memset_device_async(pimpl->pe(), spl_buf<T>::ERR_HISTO_LEN, 0, stream);
  pimpl->buf_outlier2->reset_num(stream);
  if (not pimpl->bound) return;
  auto const reset_hf = [&](auto* hf) {
    if (not hf) return;
    if (hf->hist_d()) memset_device_async(hf->hist_d(), max_bklen, 0, stream);
    hf->reset(stream);
  };
  if (psz::_2609::needs_eq4(pimpl->ppl))
    reset_hf(pimpl->buf_hfr.get());
  else {
    if (pimpl->buf_fzg) pimpl->buf_fzg->reset(stream);
    reset_hf(pimpl->buf_hf.get());
  }
}

COMPBUF_IMPL(void)::lc_wire_encoded(BYTE* external)
{
  if (pimpl->buf_lc1) pimpl->buf_lc1->wire_encoded(external);
}

COMPBUF_IMPL(void)::clear_top1() { memset_device(pimpl->top1(), pimpl->len_top1); }

// getters: array
template <typename T>
template <typename E>
E* Buf_Comp<T>::eq_d() const
{ return pimpl->template eq<E>(); }
COMPBUF_IMPL(psz_len)::eq_len3() const { return len; }
COMPBUF_IMPL(T*)::decode_fused_d() const
{ return pimpl->decode_fused() ? pimpl->decode_fused() : pimpl->d_decode_fused.get(); }
COMPBUF_IMPL(size_t)::eq_len() const { return pimpl->eq_len(); }
COMPBUF_IMPL(void)::alloc_decode_fused()  // FIXME: bin_pred reconstructs on a compress-side buf.
{
  if (not pimpl->decode_fused() and not pimpl->d_decode_fused)
    pimpl->d_decode_fused = MAKE_UNIQUE_DEVICE(T, pimpl->eq_len());
}
using psz::OutlierCell;
template <typename T>
template <typename E>
OutlierCell* Buf_Comp<T>::block_outliers_d() const
{
  auto const eq = pimpl->template eq<E>();
  return eq ? (OutlierCell*)(eq + pimpl->eq_len()) : nullptr;
}

COMPBUF_IMPL(Freq*)::top1_d() const { return pimpl->top1(); }
COMPBUF_IMPL(Freq*)::top1_h() const
{
  memcpy_allkinds<D2H>(pimpl->buf_lrz->h_top1.get(), pimpl->top1(), pimpl->len_top1);
  return pimpl->buf_lrz->h_top1.get();
}

COMPBUF_IMPL(size_t)::top1_nblk() const { return pimpl->len_top1; }

COMPBUF_IMPL(BYTE*)::compressed_d() const { return pimpl->archive(); }
COMPBUF_IMPL(BYTE*)::compressed_h() const { return pimpl->h_compressed.get(); }
COMPBUF_IMPL(size_t)::compressed_max_bytes() const { return pimpl->archive_capacity; }

COMPBUF_IMPL(size_t)::compressed_max_bytes(psz_len len, int nstage, bool eq4)
{
  if (nstage < 2) return 0;
  size_t const len_linear = (size_t)len.x * len.y * len.z;
  size_t const hf_len = hf_len_of(len);
  size_t const u2_bytes = len_linear * sizeof(u2), u4_bytes = len_linear * sizeof(u4);

  size_t overhead = Buf_HF::archive_max_words(hf_len, max_bklen, false) * sizeof(u4) - u2_bytes;
  overhead = std::max(overhead, Buf_FZG::archive_bytes(set_eq_padded(len)) - u2_bytes);
  overhead = std::max(
      overhead,
      Buf_LC::encoded_capacity(hf_len * sizeof(u2), Buf_LC::needs_align8(LC_TCMS)) - u2_bytes);
  if (eq4)
    overhead = std::max(
        overhead, Buf_HFR::archive_max_words(hf_len, max_bklen, true) * sizeof(u4) - u4_bytes);
  overhead += _2609::pad8(sizeof(psz_header));
  overhead += spl_buf<T>::anchor_cap(len) * sizeof(T);
  overhead += (_2609::seg_pass1_end - _2609::seg_encoded) * (_2609::pad8(1) - 1);

  size_t const bytes = len_linear * sizeof(T) + overhead;
  return nstage >= 3 ? Buf_LC::encoded_capacity(bytes, Buf_LC::needs_align8(LC_RTR)) : bytes;
}

COMPBUF_IMPL(void*)::outlier2_validx_d() const { return pimpl->buf_outlier2->val_idx_d(); }
COMPBUF_IMPL(M)::outlier2_host_get_num() const { return pimpl->buf_outlier2->host_get_num(); }
COMPBUF_IMPL(size_t)::outlier2_max_allowed_num() const
{ return pimpl->buf_outlier2->max_allowed_num(); }

COMPBUF_IMPL(T*)::anchor_d() const { return pimpl->anchor(); }
COMPBUF_IMPL(size_t)::anchor_len() const
{
  auto a = anchor_len3();
  return (size_t)a.x * a.y * a.z;
}
COMPBUF_IMPL(psz_len)::anchor_len3() const { return pimpl->buf_spl->anchor_len3(); }

COMPBUF_IMPL(void)::set_predictor(psz_predictor p) { pimpl->set_predictor(p); }

COMPBUF_IMPL(T*)::profiled_errors_d() const { return pimpl->pe(); };
COMPBUF_IMPL(T*)::profiled_errors_h() const { return pimpl->buf_spl->h_pe.get(); };
COMPBUF_IMPL(M)::profiled_errors_len() const { return spl_buf<T>::ERR_HISTO_LEN; };

template <typename T>
template <typename E>
u4* Buf_Comp<T>::pbk_headers_d() const
{
  if constexpr (sizeof(E) == 4)
    return pimpl->buf_hfr ? (u4*)pimpl->buf_hfr->pbk_headers_d() : nullptr;
  else
    return pimpl->buf_hf ? (u4*)pimpl->buf_hf->pbk_headers_d() : nullptr;
}
template <typename T>
template <typename E>
u1* Buf_Comp<T>::incomp_flag_d() const
{
  if constexpr (sizeof(E) == 4)
    return pimpl->buf_hfr ? pimpl->buf_hfr->incomp_flag_d() : nullptr;
  else
    return pimpl->buf_hf ? pimpl->buf_hf->incomp_flag_d() : nullptr;
}

// template <typename T>
// using Buf_Outlier = _ptb::compact_gpu<T>;

template <typename T>
using Buf_Outlier2 = _ptb::compact_GPU_DRAM2<T, M>;

using Buf_HF = phf::Buf<u2>;
using Buf_HFR = phf::Buf_HFR<u4>;
using Buf_LC = LC_Buf;
using Buf_FZG = fzg::Buf2;

COMPBUF_IMPL(Buf_Outlier2<T>*)::buf_outlier2() const { return pimpl->buf_outlier2.get(); }

COMPBUF_IMPL(Buf_HF*)::buf_hf() const { return pimpl->buf_hf.get(); }
COMPBUF_IMPL(Buf_HFR*)::buf_hfr() const { return pimpl->buf_hfr.get(); }
COMPBUF_IMPL(Buf_LC*)::buf_lc1() const { return pimpl->buf_lc1.get(); }
COMPBUF_IMPL(Buf_LC*)::buf_lc2() const { return pimpl->buf_lc2.get(); }
COMPBUF_IMPL(Buf_FZG*)::buf_fzg() const { return pimpl->buf_fzg.get(); }

}  // namespace psz

// instantiation
template class psz::Buf_Comp<f4>;
template class psz::Buf_Comp<f8>;

#define BUF_COMP_EQ(T, E)                                              \
  template E* psz::Buf_Comp<T>::eq_d<E>() const;                       \
  template psz::OutlierCell* psz::Buf_Comp<T>::block_outliers_d<E>() const; \
  template u4* psz::Buf_Comp<T>::pbk_headers_d<E>() const;             \
  template u1* psz::Buf_Comp<T>::incomp_flag_d<E>() const;

BUF_COMP_EQ(f4, u1)
BUF_COMP_EQ(f4, u2)
BUF_COMP_EQ(f4, u4)
BUF_COMP_EQ(f8, u1)
BUF_COMP_EQ(f8, u2)
BUF_COMP_EQ(f8, u4)
