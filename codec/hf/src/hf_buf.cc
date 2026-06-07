#include "hf_buf.hh"

#include <cuda.h>

#include <cstdlib>

#include "_future/scan_lookback.hh"
#include "hf.h"
#include "hfr.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "mem/gpu_event.hh"

namespace {
struct phf_eager_module_loading_init {
  phf_eager_module_loading_init() { setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0); }
};
phf_eager_module_loading_init _phf_eager_module_loading_init_singleton;
}  // namespace

using H4 = u4;
using M = PHF_METADATA;

namespace phf::_dummy {
void launch();
}

namespace phf {

template <typename E>
struct Buf<E>::impl {
  // types
  using H4 = u4;
  using M = PHF_METADATA;
  using SYM = E;
  using Header = phf_header;

  // helper struct(s)
  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  // vars
  const size_t len;
  const size_t bklen;
  const size_t rvbk4_bytes;
  const size_t bitstream_max_len;
  size_t pardeg;
  size_t sublen;
  bool use_HFR;
  bool omit_runtime_rvbk = false;  // exclude runtime rvbk from the archive
  bool use_hf_rev2_header = false;
  bool use_pbkgo = false;
  u2 rt_bklen;
  int numSMs;
  int pbkgo_max_blocks_per_sm;
  int pbkgo_max_resident_blocks;

  // internal arrays: data
  GPU_unique_dptr<H4[]> d_scratch4;
  GPU_unique_hptr<H4[]> h_scratch4;
  PHF_BYTE* d_encoded;
  PHF_BYTE* h_encoded;
  GPU_unique_dptr<H4[]> d_bitstream4;
  GPU_unique_hptr<H4[]> h_bitstream4;

  GPU_unique_dptr<H4[]> d_book4;
  GPU_unique_hptr<H4[]> h_book4;
  GPU_unique_dptr<PHF_BYTE[]> d_rvbk4;
  GPU_unique_hptr<PHF_BYTE[]> h_rvbk4;

  // internal arrays: metadata for data partitions
  GPU_unique_dptr<M[]> d_par_nbit;
  GPU_unique_hptr<M[]> h_par_nbit;
  GPU_unique_dptr<M[]> d_par_ncell;
  GPU_unique_hptr<M[]> h_par_ncell;
  GPU_unique_dptr<M[]> d_par_entry;
  GPU_unique_hptr<M[]> h_par_entry;

  // ReVISIT break-cell buffers (post-breaks_t adoption)
  using BreakCell = psz::HFR_PBK_Breaks<128>;
  GPU_unique_dptr<BreakCell[]> d_sp_breaks;
  GPU_unique_dptr<u4[]> d_sp_count;  // single-element global atomic counter
  GPU_unique_hptr<u4[]> h_sp_count;
  GPU_unique_dptr<u4[]> d_par_brnum;  // per-block break count, size = pardeg
  GPU_unique_hptr<u4[]> h_par_brnum;
  GPU_unique_dptr<u4[]> d_par_broffset;  // per-block start offset in d_sp_breaks
  GPU_unique_hptr<u4[]> h_par_broffset;
  GPU_unique_dptr<u1[]> d_par_encid;  // per-block enc_id: 0=normal, 1=incomp
  GPU_unique_hptr<u1[]> h_par_encid;

  // scan state for concat.
  GPU_unique_dptr<u4[]> d_scan_partial_aggregate;
  GPU_unique_dptr<u4[]> d_scan_incl_prefix;
  GPU_unique_dptr<int[]> d_scan_tile_status;
  int scan_num_tiles_;

  // HFR-PBK scratch (per-block headers, packed bitstream, total-ncell sink).
  using BHeader = psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>;
  GPU_unique_dptr<BHeader[]> d_pbk_headers;
  GPU_unique_hptr<BHeader[]> h_pbk_headers;
  GPU_unique_dptr<H4[]> d_packed;
  GPU_unique_dptr<u4[]> d_total_ncell;
  GPU_unique_dptr<u4[]> d_total_nbit;          // reduce_total_nbit sink
  GPU_unique_dptr<u4[]> d_pbk_packed_headers;  // 2 u4 per block; HFR family only
  GPU_unique_dptr<u4[]> d_pbkgo_state;         // 1 u4 / block; PBKGO scan state
  GPU_unique_dptr<u4[]> d_hf_rev2_header;      // bheader_backport[] = 2 u4 / block; HF_rev2 only

  // per-buf-lifetime; avoid per-encode create/destroy
  _ptb::gpu_event timing_events[3];

  // internal functions
  int _rvbk4_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 4, sizeof(SYM)); }
  int _rvbk8_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 8, sizeof(SYM)); }

  // constructor
  impl(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug) :
      len(inlen),
      bklen(_bklen),
      // HFR-PBK: per-block stride = BlockSize + MaxNumBreaks words.
      bitstream_max_len(
          _use_HFR ? (((inlen - 1) / psz::HFR_PBK_Constants::BlockSize + 1) *
                      psz::HFR_PBK_Constants::StridePerBlockWords)
                   : inlen / 2),
      use_HFR(_use_HFR),
      rvbk4_bytes(_rvbk4_bytes(_bklen))
  {
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    pbkgo_max_blocks_per_sm = 0;
    pbkgo_max_resident_blocks = 0;
    if constexpr (sizeof(SYM) <= 2) {  // pbkgo only instantiated for u1, u2
      pbkgo_max_blocks_per_sm =
          phf::module::HFR_PBKGO_encode<SYM, 10, 2, uint32_t, 128>::max_blocks_per_sm();
      pbkgo_max_resident_blocks = pbkgo_max_blocks_per_sm * numSMs;
    }

    // call dummy during expensive init
    phf::_dummy::launch();

    sublen = use_HFR ? 1024 : phf_coarse_tune_sublen(inlen);
    pardeg = (inlen - 1) / sublen + 1;

    h_scratch4 = MAKE_UNIQUE_HOST(H4, len);
    d_scratch4 = MAKE_UNIQUE_DEVICE(H4, len);
    h_book4 = MAKE_UNIQUE_HOST(H4, bklen);
    d_book4 = MAKE_UNIQUE_DEVICE(H4, bklen);
    h_rvbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, rvbk4_bytes);
    d_rvbk4 = MAKE_UNIQUE_DEVICE(PHF_BYTE, rvbk4_bytes);
    d_bitstream4 = MAKE_UNIQUE_DEVICE(H4, bitstream_max_len);
    h_bitstream4 = MAKE_UNIQUE_HOST(H4, bitstream_max_len);
    h_par_nbit = MAKE_UNIQUE_HOST(M, pardeg);
    d_par_nbit = MAKE_UNIQUE_DEVICE(M, pardeg);
    h_par_ncell = MAKE_UNIQUE_HOST(M, pardeg);
    d_par_ncell = MAKE_UNIQUE_DEVICE(M, pardeg);
    h_par_entry = MAKE_UNIQUE_HOST(M, pardeg);
    d_par_entry = MAKE_UNIQUE_DEVICE(M, pardeg);

    // Break-cell buffers (capacity heuristic ~ len/10).
    auto sp_cap = 100 + len / 10 + 1;
    d_sp_breaks = MAKE_UNIQUE_DEVICE(BreakCell, sp_cap);
    d_sp_count = MAKE_UNIQUE_DEVICE(u4, 1);
    h_sp_count = MAKE_UNIQUE_HOST(u4, 1);
    d_par_brnum = MAKE_UNIQUE_DEVICE(u4, pardeg);
    h_par_brnum = MAKE_UNIQUE_HOST(u4, pardeg);
    d_par_broffset = MAKE_UNIQUE_DEVICE(u4, pardeg);
    h_par_broffset = MAKE_UNIQUE_HOST(u4, pardeg);
    d_par_encid = MAKE_UNIQUE_DEVICE(u1, pardeg);
    h_par_encid = MAKE_UNIQUE_HOST(u1, pardeg);

    {
      using namespace psz::scan_lookback;
      scan_num_tiles_ = (int)(((size_t)pardeg + TILE_SIZE_HOST - 1) / TILE_SIZE_HOST);
      d_scan_partial_aggregate = MAKE_UNIQUE_DEVICE(u4, scan_num_tiles_ + 1);
      d_scan_incl_prefix = MAKE_UNIQUE_DEVICE(u4, scan_num_tiles_ + 1);
      d_scan_tile_status = MAKE_UNIQUE_DEVICE(int, scan_num_tiles_ + 1);
    }

    // total_ncell / total_nbit sinks (1 u4 each; also used by Huffman_rev1).
    d_total_ncell = MAKE_UNIQUE_DEVICE(u4, 1);
    d_total_nbit = MAKE_UNIQUE_DEVICE(u4, 1);
    // HF_rev2 AoS bheader_backport[] (always allocated; 8 B/block).
    d_hf_rev2_header = MAKE_UNIQUE_DEVICE(u4, 2 * pardeg);

    if (use_HFR) {
      using K = psz::HFR_PBK_Constants;
      d_pbk_headers = MAKE_UNIQUE_DEVICE(BHeader, pardeg);
      h_pbk_headers = MAKE_UNIQUE_HOST(BHeader, pardeg);
      d_packed = MAKE_UNIQUE_DEVICE(H4, pardeg * K::BlockSize);
      d_pbk_packed_headers = MAKE_UNIQUE_DEVICE(u4, 2 * pardeg);  // 2 u4 / block
      d_pbkgo_state = MAKE_UNIQUE_DEVICE(u4, pardeg);             // init to 0 (INVALID)
      memset_device(d_pbkgo_state.get(), pardeg);
    }

    // repurpose scratch after several substeps
    d_encoded = (u1*)d_scratch4.get();
    h_encoded = (u1*)h_scratch4.get();

    // Init scan state once at buf init (per-encode init is the caller's reset()).
    psz::scan_lookback::launch_init_host(
        d_scan_partial_aggregate.get(), d_scan_incl_prefix.get(), d_scan_tile_status.get(),
        scan_num_tiles_, /*stream*/ 0);
    cudaDeviceSynchronize();

    for (int i = 0; i < 3; ++i) timing_events[i] = _ptb::make_gpu_event();
  }

  // public functions
  void memcpy_merge(Header& header, phf_stream_t stream)

  {
    auto memcpy_start = d_encoded;
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _rvbk{d_rvbk4.get(), rvbk4_bytes, header.entry[PHFHEADER_RVBK]};
    // SoA per-block metadata is for HF / HF-rev1 only.
    const bool _skip_soa_meta = use_HFR or use_hf_rev2_header;
    memcpy_helper _par_nbit{
        d_par_nbit.get(), (_skip_soa_meta ? size_t{0} : pardeg * sizeof(M)),
        header.entry[PHFHEADER_PAR_NBIT]};
    memcpy_helper _par_entry{
        d_par_entry.get(), (_skip_soa_meta ? size_t{0} : pardeg * sizeof(M)),
        header.entry[PHFHEADER_PAR_ENTRY]};
    // HFR/PBKC read the compact bitstream straight from d_packed.
    H4* bitstream_src =
        use_pbkgo ? d_bitstream4.get() : (use_HFR ? d_packed.get() : d_bitstream4.get());
    memcpy_helper _bitstream{
        bitstream_src, (size_t)header.total_ncell * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
      if (var.nbyte == 0) return;  // skip dead sections (PBKC's rvbk, brnum=0 sections)
      cudaMemcpyAsync(
          start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    };

    cudaMemcpyAsync(start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream);

    // RVBK: omitted for PBKC (uses prebuilt pbk25_r128 constant on decode side).
    if (not omit_runtime_rvbk) d2d_memcpy_merge(_rvbk);
    d2d_memcpy_merge(_par_nbit);
    d2d_memcpy_merge(_par_entry);
    d2d_memcpy_merge(_bitstream);

    // HF_rev2 — AoS bheader_backport[] in PHFHEADER_HF_REV2_HEADER.
    if (use_hf_rev2_header) {
      memcpy_helper _hf_rev2_header{
          d_hf_rev2_header.get(), 2 * pardeg * sizeof(u4), header.entry[PHFHEADER_HF_REV2_HEADER]};
      d2d_memcpy_merge(_hf_rev2_header);
    }

    // HFR family — packed per-block headers (brnum sections only if brnum>0).
    if (use_HFR) {
      memcpy_helper _pbk_headers{
          d_pbk_packed_headers.get(), 2 * pardeg * sizeof(u4),
          header.entry[PHFHEADER_PBK_HEADERS]};
      d2d_memcpy_merge(_pbk_headers);
      if (header.brnum > 0) {
        memcpy_helper _par_brnum{
            d_par_brnum.get(), pardeg * sizeof(u4), header.entry[PHFHEADER_PAR_BRNUM]};
        memcpy_helper _sp_breaks{
            d_sp_breaks.get(), header.brnum * sizeof(BreakCell),
            header.entry[PHFHEADER_SP_BREAKS]};
        d2d_memcpy_merge(_par_brnum);
        d2d_memcpy_merge(_sp_breaks);
      }
    }
  }

  void clear_buffer()
  {
    memset_device(d_scratch4.get(), len);
    memset_device(d_book4.get(), bklen);
    memset_device(d_rvbk4.get(), rvbk4_bytes);
    memset_device(d_bitstream4.get(), bitstream_max_len);
    memset_device(d_par_nbit.get(), pardeg);
    memset_device(d_par_ncell.get(), pardeg);
    memset_device(d_par_entry.get(), pardeg);
  }
};

#define PHF_BUF_DEF(...) \
  template <typename E>  \
  __VA_ARGS__ phf::Buf<E>

PHF_BUF_DEF()::Buf(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug) :
    pimpl(std::make_unique<impl>(inlen, _bklen, _pardeg, _use_HFR, debug))
{
}

PHF_BUF_DEF()::~Buf() {}

// a series of getters: variables
PHF_BUF_DEF(size_t)::rvbk_bytes() const { return pimpl->rvbk4_bytes; }
PHF_BUF_DEF(u2)::rt_bklen() const { return pimpl->rt_bklen; }
PHF_BUF_DEF(int)::numSMs() const { return pimpl->numSMs; }
PHF_BUF_DEF(int)::pbkgo_max_blocks_per_sm() const { return pimpl->pbkgo_max_blocks_per_sm; }
PHF_BUF_DEF(int)::pbkgo_max_resident_blocks() const { return pimpl->pbkgo_max_resident_blocks; }
PHF_BUF_DEF(size_t)::sublen() const { return pimpl->sublen; }
PHF_BUF_DEF(size_t)::pardeg() const { return pimpl->pardeg; }
PHF_BUF_DEF(size_t)::bitstream_max_len() const { return pimpl->bitstream_max_len; }

// a series of getters: arrays
PHF_BUF_DEF(H4*)::book_d() const { return pimpl->d_book4.get(); }
PHF_BUF_DEF(H4*)::book_h() const { return pimpl->h_book4.get(); }
PHF_BUF_DEF(u1*)::rvbk_d() const { return pimpl->d_rvbk4.get(); }
PHF_BUF_DEF(u1*)::rvbk_h() const { return pimpl->h_rvbk4.get(); }
PHF_BUF_DEF(H4*)::scratch_d() const { return pimpl->d_scratch4.get(); }
PHF_BUF_DEF(H4*)::scratch_h() const { return pimpl->h_scratch4.get(); }
PHF_BUF_DEF(M*)::par_nbit_d() const { return pimpl->d_par_nbit.get(); }
PHF_BUF_DEF(M*)::par_nbit_h() const { return pimpl->h_par_nbit.get(); }
PHF_BUF_DEF(M*)::par_ncell_d() const { return pimpl->d_par_ncell.get(); }
PHF_BUF_DEF(M*)::par_ncell_h() const { return pimpl->h_par_ncell.get(); }
PHF_BUF_DEF(M*)::par_entry_d() const { return pimpl->d_par_entry.get(); }
PHF_BUF_DEF(M*)::par_entry_h() const { return pimpl->h_par_entry.get(); }
PHF_BUF_DEF(H4*)::bitstream_d() const { return pimpl->d_bitstream4.get(); }
PHF_BUF_DEF(H4*)::bitstream_h() const { return pimpl->h_bitstream4.get(); }
PHF_BUF_DEF(PHF_BYTE*)::encoded_d() const { return pimpl->d_encoded; }
PHF_BUF_DEF(PHF_BYTE*)::encoded_h() const { return pimpl->h_encoded; }

PHF_BUF_DEF(u4*)::scan_partial_aggregate_d() const
{
  return pimpl->d_scan_partial_aggregate.get();
}
PHF_BUF_DEF(u4*)::scan_incl_prefix_d() const { return pimpl->d_scan_incl_prefix.get(); }
PHF_BUF_DEF(int*)::scan_tile_status_d() const { return pimpl->d_scan_tile_status.get(); }
PHF_BUF_DEF(int)::scan_num_tiles() const { return pimpl->scan_num_tiles_; }

// method
PHF_BUF_DEF(void)::update_header(phf_header& header)
{
  header.bklen = pimpl->rt_bklen;
  header.sublen = pimpl->sublen;
  header.pardeg = pimpl->pardeg;
  header.original_len = pimpl->len;
  header.brnum = 0;  // inline-breaks: per-block n_breaks lives in the bheader array.
}

PHF_BUF_DEF(void)::calc_offset(phf_header& header, M* byte_offsets)
{
  byte_offsets[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
  // RVBK omitted for PBKC.
  byte_offsets[PHFHEADER_RVBK] = pimpl->omit_runtime_rvbk ? 0 : rvbk_bytes();
  // PAR_NBIT / PAR_ENTRY: HF / HF-rev1 only.
  const bool _skip_soa_meta = pimpl->use_HFR or pimpl->use_hf_rev2_header;
  byte_offsets[PHFHEADER_PAR_NBIT] = _skip_soa_meta ? 0 : pimpl->pardeg * sizeof(M);
  byte_offsets[PHFHEADER_PAR_ENTRY] = _skip_soa_meta ? 0 : pimpl->pardeg * sizeof(M);
  byte_offsets[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;
  // brnum-bearing sections: legacy detached-breaks only (brnum>0).
  // TODO PAR_BROFFSET derivable from PAR_BRNUM; wire up when brnum>0 is exercised.
  byte_offsets[PHFHEADER_PAR_BRNUM] =
      (pimpl->use_HFR and header.brnum > 0) ? pimpl->pardeg * sizeof(u4) : 0;
  byte_offsets[PHFHEADER_PAR_BROFFSET] = 0;
  byte_offsets[PHFHEADER_SP_BREAKS] =
      (pimpl->use_HFR and header.brnum > 0) ? header.brnum * sizeof(typename impl::BreakCell) : 0;
  byte_offsets[PHFHEADER_PAR_ENCID] = 0;  // superseded by PBK_HEADERS
  byte_offsets[PHFHEADER_PBK_HEADERS] = pimpl->use_HFR ? 2 * pimpl->pardeg * sizeof(u4) : 0;
  byte_offsets[PHFHEADER_HF_REV2_HEADER] =
      pimpl->use_hf_rev2_header ? 2 * pimpl->pardeg * sizeof(u4) : 0;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = byte_offsets[i - 1];
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
}

PHF_BUF_DEF(void)::set_omit_runtime_rvbk(bool v) { pimpl->omit_runtime_rvbk = v; }
PHF_BUF_DEF(void)::set_use_hf_rev2_header(bool v) { pimpl->use_hf_rev2_header = v; }
PHF_BUF_DEF(void)::set_use_pbkgo(bool v) { pimpl->use_pbkgo = v; }
PHF_BUF_DEF(void*)::timing_event(int idx) const { return (void*)pimpl->timing_events[idx].get(); }

// method: set internal variable
PHF_BUF_DEF(void)::register_runtime_bklen(const int _rt_bklen) { pimpl->rt_bklen = _rt_bklen; }

// HFR break-cell accessors
PHF_BUF_DEF(psz::HFR_PBK_Breaks<128>*)::sp_breaks_d() const { return pimpl->d_sp_breaks.get(); }
PHF_BUF_DEF(u4*)::sp_count_d() const { return pimpl->d_sp_count.get(); }
PHF_BUF_DEF(u4*)::par_brnum_d() const { return pimpl->d_par_brnum.get(); }
PHF_BUF_DEF(u4*)::par_brnum_h() const { return pimpl->h_par_brnum.get(); }
PHF_BUF_DEF(u4*)::par_broffset_d() const { return pimpl->d_par_broffset.get(); }
PHF_BUF_DEF(u4*)::par_broffset_h() const { return pimpl->h_par_broffset.get(); }
PHF_BUF_DEF(u1*)::par_encid_d() const { return pimpl->d_par_encid.get(); }
PHF_BUF_DEF(u1*)::par_encid_h() const { return pimpl->h_par_encid.get(); }

PHF_BUF_DEF(void)::memcpy_merge(phf_header& header, phf_stream_t stream)
{
  pimpl->memcpy_merge(header, stream);
}

// method, same-name
PHF_BUF_DEF(void)::clear_buffer() { pimpl->clear_buffer(); }

// Per-encode reset: scan-state init for LAGO.
PHF_BUF_DEF(void)::reset(phf_stream_t stream)
{
  psz::scan_lookback::launch_init_host(
      pimpl->d_scan_partial_aggregate.get(), pimpl->d_scan_incl_prefix.get(),
      pimpl->d_scan_tile_status.get(), pimpl->scan_num_tiles_, stream);
}

PHF_BUF_DEF(void)::reset_HFR(phf_stream_t stream)
{
  // Per-encode pay-forward reset for HFR-family scan / lookback state.
  reset(stream);
  if (pimpl->d_pbkgo_state)
    cudaMemsetAsync(
        pimpl->d_pbkgo_state.get(), 0, pimpl->pardeg * sizeof(u4), (cudaStream_t)stream);
}

PHF_BUF_DEF(psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>*)::pbk_headers_d() const
{
  return pimpl->d_pbk_headers.get();
}
PHF_BUF_DEF(psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>*)::pbk_headers_h() const
{
  return pimpl->h_pbk_headers.get();
}
PHF_BUF_DEF(u4*)::packed_d() const { return pimpl->d_packed.get(); }
PHF_BUF_DEF(u4*)::total_ncell_d() const { return pimpl->d_total_ncell.get(); }
PHF_BUF_DEF(u4*)::total_nbit_d() const { return pimpl->d_total_nbit.get(); }
PHF_BUF_DEF(u4*)::pbk_packed_headers_d() const { return pimpl->d_pbk_packed_headers.get(); }
PHF_BUF_DEF(u4*)::pbkgo_state_d() const { return pimpl->d_pbkgo_state.get(); }
PHF_BUF_DEF(u4*)::hf_rev2_header_d() const { return pimpl->d_hf_rev2_header.get(); }

}  // namespace phf

template struct phf::Buf<u1>;
template struct phf::Buf<u2>;
template struct phf::Buf<u4>;

#undef PHF_BUF_DEF