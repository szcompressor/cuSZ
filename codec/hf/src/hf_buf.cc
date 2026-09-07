#include "hf_buf.hh"

#include <cuda.h>

#include <cstddef>
#include <cstdlib>

#include "_future/scan_lookback.hh"
#include "hf.h"
#include "hfd26.hh"
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

extern "C" void* pbk25_r128_rvbk_d_ptr();  // pbk25_r128_d.cu

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
  bool use_prebuilt_rvbk = false;  // exclude runtime RVBK from the archive
  bool use_pbkgo = false;
  bool use_global_encid = false;  // HFR-v3: record global PBK ID
  bool lut_ready = false;         // HFD26 LUT cache
  u2 rt_bklen;
  int num_sms;
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

  // scan state for concat.
  GPU_unique_dptr<u4[]> d_scan_partial_aggregate;
  GPU_unique_dptr<u4[]> d_scan_incl_prefix;
  GPU_unique_dptr<int[]> d_scan_tile_status;
  int scan_num_tiles_;

  // HFR-PBK scratch
  using BHeader = psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>;
  GPU_unique_dptr<BHeader[]> d_pbk_headers;
  GPU_unique_hptr<BHeader[]> h_pbk_headers;
  GPU_unique_dptr<u1[]> d_incomp_flag;  // [pardeg] decode message: 1 = unpred-incomp block
  GPU_unique_dptr<H4[]> d_packed;
  GPU_unique_dptr<u4[]> d_total_ncell;
  GPU_unique_dptr<u4[]> d_pick_encid;          // HFR-v3: 1-word global PBK id from the pick kernel
  GPU_unique_dptr<u4[]> d_pbk_packed_headers;  // two u4 per block
  GPU_unique_dptr<u4[]> d_pbkgo_state;         // one u4 per block
  GPU_unique_dptr<phf::LutEntry[]> d_lut;  // HFD26: NumBooks x 256 LutEntry

  // per-buf-lifetime; avoid per-encode create/destroy
  _ptb::gpu_event timing_events[3];

  // FIXME: may duplicate somewhere.
  static size_t archive_max_words(size_t len, size_t rvbk_bytes)
  {
    const size_t nblock = (len - 1) / psz::HFR_PBK_Constants::BlockSize + 1;
    return (sizeof(Header) + rvbk_bytes + sizeof(H4) - 1) / sizeof(H4) +
           (nblock * psz::HFR_PBK_Constants::StridePerBlockWords * sizeof(SYM) + sizeof(H4) - 1) /
               sizeof(H4) +
           1;
  }

  // internal functions
  int _rvbk4_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 4, sizeof(SYM)); }
  int _rvbk8_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 8, sizeof(SYM)); }

  // constructor
  impl(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug, bool use_sublen_1ki) :
      len(inlen),
      bklen(_bklen),
      // per-block staging stride, padded to the M12 worst case (E=u4 dense ~= 1 word/symbol).
      bitstream_max_len(
          _use_HFR ? (((inlen - 1) / psz::HFR_PBK_Constants::BlockSize + 1) *
                          psz::HFR_PBK_Constants::StridePerBlockWords +
                      ((inlen - 1) / psz::HFR_PBK_Constants::BlockSize + 1) *
                          (psz::HFR_PBK_C12::MaxUnpredWords + psz::HFR_PBK_C12::MaxNumBreaks))
                   : inlen / 2),
      use_HFR(_use_HFR),
      rvbk4_bytes(_rvbk4_bytes(_bklen))
  {
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    pbkgo_max_blocks_per_sm =
        phf::module::HFR_PBKGO_encode<SYM, 10, 2, uint32_t, 128>::max_blocks_per_sm();
    pbkgo_max_resident_blocks = pbkgo_max_blocks_per_sm * num_sms;

    // call dummy during expensive init
    phf::_dummy::launch();

    sublen = (use_HFR or use_sublen_1ki) ? 1024 : phf_coarse_tune_sublen(inlen);
    pardeg = (inlen - 1) / sublen + 1;

    const size_t hfr_incomp_pad =
        use_HFR ? pardeg * (psz::HFR_PBK_C12::MaxUnpredWords + psz::HFR_PBK_C12::MaxNumBreaks) : 0;

    const size_t scratch_bytes = use_HFR ? archive_max_words(len, rvbk4_bytes) * sizeof(H4)
                                         : sizeof(H4) * (len + hfr_incomp_pad);
    d_scratch4 = MAKE_UNIQUE_DEVICE(H4, (scratch_bytes + sizeof(H4) - 1) / sizeof(H4));
    h_book4 = MAKE_UNIQUE_HOST(H4, bklen);
    d_book4 = MAKE_UNIQUE_DEVICE(H4, bklen);
    h_rvbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, rvbk4_bytes);
    d_rvbk4 = MAKE_UNIQUE_DEVICE(PHF_BYTE, rvbk4_bytes);
    const size_t bitstream_bytes = sizeof(SYM) * bitstream_max_len;
    d_bitstream4 = MAKE_UNIQUE_DEVICE(H4, (bitstream_bytes + sizeof(H4) - 1) / sizeof(H4));
    d_par_nbit = MAKE_UNIQUE_DEVICE(M, pardeg);
    d_par_ncell = MAKE_UNIQUE_DEVICE(M, pardeg);
    h_par_entry = MAKE_UNIQUE_HOST(M, pardeg);
    d_par_entry = MAKE_UNIQUE_DEVICE(M, pardeg);

    {
      using namespace psz::scan_lookback;
      scan_num_tiles_ = (int)(((size_t)pardeg + TILE_SIZE_HOST - 1) / TILE_SIZE_HOST);
      d_scan_partial_aggregate = MAKE_UNIQUE_DEVICE(u4, scan_num_tiles_ + 1);
      d_scan_incl_prefix = MAKE_UNIQUE_DEVICE(u4, scan_num_tiles_ + 1);
      d_scan_tile_status = MAKE_UNIQUE_DEVICE(int, scan_num_tiles_ + 1);
    }

    d_total_ncell = MAKE_UNIQUE_DEVICE(u4, 1);
    d_pick_encid = MAKE_UNIQUE_DEVICE(u4, 1);

    const auto nblock_1ki = (len - 1) / psz::HFR_PBK_Constants::BlockSize + 1;
    // bheader AoS: nblock_1ki per-block for the HFR family, pardeg per-chunk for HF / HF-rev2.
    const auto n_bheaders = nblock_1ki > pardeg ? nblock_1ki : pardeg;
    d_pbk_headers = MAKE_UNIQUE_DEVICE(BHeader, n_bheaders);

    if (use_HFR) {
      using K = psz::HFR_PBK_Constants;
      d_incomp_flag = MAKE_UNIQUE_DEVICE(u1, pardeg);  // 0 = normal; 1 = use incomp-31
      const size_t packed_bytes = sizeof(SYM) * (pardeg * K::BlockSize + hfr_incomp_pad);
      d_packed = MAKE_UNIQUE_DEVICE(H4, (packed_bytes + sizeof(H4) - 1) / sizeof(H4));
      d_pbk_packed_headers = MAKE_UNIQUE_DEVICE(u4, 2 * pardeg);  // two u4 per block
      d_pbkgo_state = MAKE_UNIQUE_DEVICE(u4, pardeg);             // init to 0 (INVALID)

      d_lut = MAKE_UNIQUE_DEVICE(phf::LutEntry, K::NumBooks * 256);

      phf::module::HFD26<SYM, u4, u1>::build_lut(
          (u1*)pbk25_r128_rvbk_d_ptr(), (int)K::RvbkBytesPerBook, (int)K::NumBooks,
          d_lut.get(), /*stream*/ 0);
    }

    // repurpose scratch after several substeps
    d_encoded = (u1*)d_scratch4.get();
    h_encoded = nullptr;  // no pinned mirror; encoded_h() has no callers

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
    // SoA metadata retired: the per-block header ships as bheader AoS.
    memcpy_helper _par_nbit{d_par_nbit.get(), size_t{0}, header.entry[PHFHEADER_PAR_NBIT]};
    memcpy_helper _par_entry{d_par_entry.get(), size_t{0}, header.entry[PHFHEADER_PAR_ENTRY]};
    // HFR/PBKC read the compact bitstream straight from d_packed.
    H4* bitstream_src =
        use_pbkgo ? d_bitstream4.get() : (use_HFR ? d_packed.get() : d_bitstream4.get());
    memcpy_helper _bitstream{
        bitstream_src, (size_t)header.total_ncell * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
      if (var.nbyte == 0) return;  // skip dead sections (PBKC's RVBK, brnum=0 sections)
      cudaMemcpyAsync(
          start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    };

    cudaMemcpyAsync(start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream);

    if (use_global_encid)  // HFR-v3
      cudaMemcpyAsync(
          start + offsetof(Header, g_encid), d_pick_encid.get(), sizeof(u1),
          cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

    if (not use_prebuilt_rvbk) d2d_memcpy_merge(_rvbk);  // not applicable for PBK
    d2d_memcpy_merge(_par_nbit);
    d2d_memcpy_merge(_par_entry);
    d2d_memcpy_merge(_bitstream);

    // header.pardeg, not the buf's: 2Ki+ PBKC encodes fewer blocks than the 1Ki default.
    if (not use_HFR) {  // HF / HFr2 ship the per-block header as bheader AoS.
      memcpy_helper _hf_rev2_header{
          (u4*)d_pbk_headers.get(), 2 * (size_t)header.pardeg * sizeof(u4),
          header.entry[PHFHEADER_HF_REV2_HEADER]};
      d2d_memcpy_merge(_hf_rev2_header);
    }

    if (use_HFR) {  // HFR family: packed per-block headers.
      memcpy_helper _pbk_headers{
          d_pbk_packed_headers.get(), 2 * (size_t)header.pardeg * sizeof(u4),
          header.entry[PHFHEADER_PBK_HEADERS]};
      d2d_memcpy_merge(_pbk_headers);
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

PHF_BUF_DEF()::Buf(
    size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug, bool use_sublen_1ki) :
    pimpl(std::make_unique<impl>(inlen, _bklen, _pardeg, _use_HFR, debug, use_sublen_1ki))
{
}

PHF_BUF_DEF()::~Buf() {}

// a series of getters: variables
PHF_BUF_DEF(size_t)::rvbk_bytes() const { return pimpl->rvbk4_bytes; }
PHF_BUF_DEF(u2)::rt_bklen() const { return pimpl->rt_bklen; }
PHF_BUF_DEF(int)::num_sms() const { return pimpl->num_sms; }
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
{ return pimpl->d_scan_partial_aggregate.get(); }
PHF_BUF_DEF(u4*)::scan_incl_prefix_d() const { return pimpl->d_scan_incl_prefix.get(); }
PHF_BUF_DEF(int*)::scan_tile_status_d() const { return pimpl->d_scan_tile_status.get(); }
PHF_BUF_DEF(int)::scan_num_tiles() const { return pimpl->scan_num_tiles_; }

// method
PHF_BUF_DEF(void)::update_header(phf_header& header)
{
  header.log_bklen = (u1)__builtin_ctz((unsigned)pimpl->rt_bklen);  // bklen is power-of-2
  header.sublen = pimpl->sublen;
  header.pardeg = pimpl->pardeg;
  header.ori_len = pimpl->len;
}

PHF_BUF_DEF(void)::calc_offset(phf_header& header, M* byte_offsets)
{
  byte_offsets[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
  // RVBK omitted for PBKC.
  byte_offsets[PHFHEADER_RVBK] = pimpl->use_prebuilt_rvbk ? 0 : rvbk_bytes();
  // SoA metadata retired: the per-block header ships as bheader AoS.
  byte_offsets[PHFHEADER_PAR_NBIT] = 0;
  byte_offsets[PHFHEADER_PAR_ENTRY] = 0;
  byte_offsets[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;
  byte_offsets[PHFHEADER_PBK_HEADERS] = pimpl->use_HFR ? 2 * header.pardeg * sizeof(u4) : 0;
  byte_offsets[PHFHEADER_HF_REV2_HEADER] = not pimpl->use_HFR ? 2 * header.pardeg * sizeof(u4) : 0;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = byte_offsets[i - 1];
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
}

PHF_BUF_DEF(void)::set_use_prebuilt_rvbk(bool v) { pimpl->use_prebuilt_rvbk = v; }
PHF_BUF_DEF(void)::set_use_pbkgo(bool v) { pimpl->use_pbkgo = v; }
PHF_BUF_DEF(void)::set_use_global_encid(bool v) { pimpl->use_global_encid = v; }
PHF_BUF_DEF(u4*)::pick_encid_d() const { return pimpl->d_pick_encid.get(); }
PHF_BUF_DEF(void*)::timing_event(int idx) const { return (void*)pimpl->timing_events[idx].get(); }

// method: set internal variable
PHF_BUF_DEF(void)::set_rt_bklen(const int _rt_bklen) { pimpl->rt_bklen = _rt_bklen; }

PHF_BUF_DEF(void)::memcpy_merge(phf_header& header, phf_stream_t stream)
{ pimpl->memcpy_merge(header, stream); }

// method, same-name
PHF_BUF_DEF(void)::clear_buffer() { pimpl->clear_buffer(); }

// Per-encode reset: scan-state init for LAGO.
PHF_BUF_DEF(void)::reset(phf_stream_t stream)
{
  pimpl->lut_ready = false;  // new encode -> LUT is stale
  psz::scan_lookback::launch_init_host(
      pimpl->d_scan_partial_aggregate.get(), pimpl->d_scan_incl_prefix.get(),
      pimpl->d_scan_tile_status.get(), pimpl->scan_num_tiles_, stream);
}

PHF_BUF_DEF(void)::reset_HFR(phf_stream_t stream)
{
  reset(stream);
  if (pimpl->d_pbkgo_state)
    cudaMemsetAsync(
        pimpl->d_pbkgo_state.get(), 0, pimpl->pardeg * sizeof(u4), (cudaStream_t)stream);
  // clear bheaders for multiple runs
  if (pimpl->d_pbk_headers)
    cudaMemsetAsync(
        pimpl->d_pbk_headers.get(), 0, pimpl->pardeg * 2 * sizeof(u4), (cudaStream_t)stream);
}

PHF_BUF_DEF(psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>*)::pbk_headers_d() const
{ return pimpl->d_pbk_headers.get(); }
PHF_BUF_DEF(psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>*)::pbk_headers_h() const
{ return pimpl->h_pbk_headers.get(); }
PHF_BUF_DEF(u4*)::packed_d() const { return pimpl->d_packed.get(); }
PHF_BUF_DEF(u4*)::total_ncell_d() const { return pimpl->d_total_ncell.get(); }
PHF_BUF_DEF(u4*)::pbk_packed_headers_d() const { return pimpl->d_pbk_packed_headers.get(); }
PHF_BUF_DEF(u1*)::incomp_flag_d() const { return pimpl->d_incomp_flag.get(); }
PHF_BUF_DEF(u4*)::pbkgo_state_d() const { return pimpl->d_pbkgo_state.get(); }
PHF_BUF_DEF(phf::LutEntry*)::lut_d() const { return pimpl->d_lut.get(); }
PHF_BUF_DEF(bool)::lut_ready() const { return pimpl->lut_ready; }
PHF_BUF_DEF(void)::lut_ready(bool v) { pimpl->lut_ready = v; }

}  // namespace phf

template struct phf::Buf<u1>;
template struct phf::Buf<u2>;
template struct phf::Buf<u4>;

#undef PHF_BUF_DEF