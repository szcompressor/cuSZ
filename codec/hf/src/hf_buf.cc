#include "hf_buf.hh"

#include <cuda.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "_future/scan_lookback.hh"
#include "hf.h"
#include "hfd26.hh"
#include "hfr.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "mem/gpu_event.hh"
#include "mem/plan.h"

namespace {
struct phf_eager_module_loading_init {
  phf_eager_module_loading_init() { setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0); }
};
phf_eager_module_loading_init _phf_eager_module_loading_init_singleton;
}  // namespace

using H4 = u4;
using M = PHF_METADATA;

extern "C" void* pbk25_r128_book_d_ptr();
extern "C" void* pbk25_r128_rvbk_d_ptr();
extern "C" void* pbk25_r128_lut_d_ptr();

namespace phf::_dummy {
void launch();
}

namespace phf {

template <typename E>
struct Buf<E>::impl : _ptb::buf_base {
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
  size_t len;
  const size_t max_inlen;
  const size_t bklen;
  const size_t rvbk4_bytes;
  const size_t bitstream_max_len;
  size_t pardeg;
  size_t sublen;
  bool use_sublen_1ki;
  bool use_HFR;
  bool is_comp;
  PHF_BYTE* archive_dst = nullptr;
  H4* pbk_book = nullptr;
  PHF_BYTE* pbk_rvbk = nullptr;
  phf::LutEntry* pbk_lut = nullptr;
  bool use_prebuilt_rvbk = false;  // exclude runtime RVBK from the archive
  bool use_pbkgo = false;
  bool use_global_encid = false;  // HFR-v3: record global PBK ID
  u2 rt_bklen;
  int num_sms;
  int pbkgo_max_blocks_per_sm;
  int pbkgo_max_resident_blocks;

  // clang-format off
  enum : int {
    SCRATCH, BOOK, RVBK, BITSTREAM, PAR_NBIT, PAR_NCELL, PAR_ENTRY,
    SCAN_PARTIAL, SCAN_INCL, SCAN_STATUS, PICK_ENCID, INCOMP_FLAG, PBKGO_STATE, LUT, HIST,
    TOTAL_NCELL, HF_HEADER,         // HF
    INFO_TOTAL_NCELL, PBK_HEADERS,  // HFR; INFO_: derivable from PBK_HEADERS
    NUM_SYM };
  // clang-format on

  using BHeader = psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>;

  GPU_unique_dptr<u1[]> d_data, d_state;

  GPU_unique_hptr<H4[]> h_scratch4, h_bitstream4, h_book4;
  GPU_unique_hptr<u4[]> h_hist;
  GPU_unique_hptr<PHF_BYTE[]> h_rvbk4;
  GPU_unique_hptr<M[]> h_par_nbit, h_par_ncell, h_par_entry;
  GPU_unique_hptr<BHeader[]> h_pbk_headers;
  PHF_BYTE* h_encoded;

  int scan_num_tiles_;

  // per-buf-lifetime; avoid per-encode create/destroy
  _ptb::gpu_event timing_events[3];

  // FIXME: may duplicate somewhere.
  static size_t archive_max_words(size_t len, size_t rvbk_bytes, bool use_HFR)
  {
    return words(PHFHEADER_FORCED_ALIGN + rvbk_bytes) + bitstream_words_for(len, use_HFR) +
           words(pardeg_cap(len, use_HFR) * sizeof(BHeader));
  }

  // internal functions
  static int _rvbk4_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 4, sizeof(SYM)); }
  static int _rvbk8_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 8, sizeof(SYM)); }

  static size_t words(size_t nbyte) { return (nbyte + sizeof(H4) - 1) / sizeof(H4); }

  static tables plan_HF(size_t inlen, size_t bklen, bool is_comp)
  {
    using K = psz::HFR_PBK_Constants;
    auto const pardeg = pardeg_cap(inlen, /*use_HFR=*/false);
    auto const rvbk_bytes = (size_t)_rvbk4_bytes((int)bklen);
    auto const nblock_1ki = (inlen - 1) / K::BlockSize + 1;
    auto const nbhdr = nblock_1ki > pardeg ? nblock_1ki : pardeg;
    auto const ntile = (size_t)tune_scan_tiles(pardeg) + 1;

    auto const enc = [is_comp](size_t n) { return is_comp ? n : 0; };

    return {
        // arrays
        {{SCRATCH, U4, enc(words(sizeof(H4) * inlen))},
         {BOOK, U4, enc(bklen)},
         {RVBK, U1, enc(rvbk_bytes)},
         {BITSTREAM, U4, enc(bitstream_words_for(inlen, false))},
         {PAR_NBIT, U4, pardeg},
         {PAR_NCELL, U4, enc(pardeg)},
         {PAR_ENTRY, U4, pardeg}},
        // metadata
        {{SCAN_PARTIAL, U4, enc(ntile)},
         {SCAN_INCL, U4, enc(ntile)},
         {SCAN_STATUS, I4, enc(ntile)},
         {TOTAL_NCELL, U4, enc(1)},
         {HIST, U4, enc(bklen)},
         {HF_HEADER, U1, enc(nbhdr * sizeof(BHeader))}}};
  }

  static tables plan_HFR(size_t inlen, size_t bklen, bool is_comp, bool external_archive)
  {
    using K = psz::HFR_PBK_Constants;
    auto const pardeg = pardeg_cap(inlen, /*use_HFR=*/true);
    // use 1Ki to allocate the upper-bound amount
    auto const nblock_1ki = (inlen - 1) / K::BlockSize + 1;
    auto const rvbk_bytes = (size_t)_rvbk4_bytes((int)bklen);
    auto const ntile = (size_t)tune_scan_tiles(pardeg) + 1;

    auto const enc = [is_comp](size_t n) { return is_comp ? n : 0; };
    auto const scratch_bytes = archive_max_words(inlen, rvbk_bytes, true) * sizeof(H4);

    return {
        // arrays
        {{SCRATCH, U4, external_archive ? 0 : enc(words(scratch_bytes))},
         {BOOK, U4, enc(bklen)},
         {RVBK, U1, enc(rvbk_bytes)},
         {BITSTREAM, U4, enc(bitstream_words_for(inlen, true))}},
        // metadata
        {{SCAN_PARTIAL, U4, enc(ntile)},
         {SCAN_INCL, U4, enc(ntile)},
         {SCAN_STATUS, I4, enc(ntile)},
         {INFO_TOTAL_NCELL, U4, enc(1)},
         {HIST, U4, enc(bklen)},
         {PICK_ENCID, U4, enc(1)},
         {PBK_HEADERS, U1, enc(nblock_1ki * sizeof(BHeader))},
         {INCOMP_FLAG, U1, pardeg},
         {PBKGO_STATE, U4, enc(pardeg)},
         {LUT, U1, is_comp ? 0 : 256 * sizeof(phf::LutEntry)}}};
  }

  static size_t pardeg_of(size_t inlen, bool use_HFR, bool use_sublen_1ki)
  { return (inlen - 1) / tune_sublen(inlen, use_HFR, use_sublen_1ki) + 1; }

  static size_t tune_sublen(size_t inlen, bool use_HFR, bool use_sublen_1ki)
  { return (use_HFR or use_sublen_1ki) ? 1024 : phf_coarse_tune_sublen(inlen); }

  static size_t pardeg_cap(size_t inlen, bool use_HFR)
  { return use_HFR ? pardeg_of(inlen, true, false) : (inlen - 1) / phf_coarse_tune_sublen(1) + 1; }

  static int tune_scan_tiles(size_t pardeg)
  {
    using namespace psz::scan_lookback;
    return (int)(((size_t)pardeg + TILE_SIZE_HOST - 1) / TILE_SIZE_HOST);
  }

  impl(
      size_t inlen, size_t _bklen, bool _use_HFR, bool use_sublen_1ki, bool _is_comp,
      void* _archive_dst, tables t) :
      buf_base(std::move(t)),
      len(inlen),
      max_inlen(inlen),
      bklen(_bklen),
      rvbk4_bytes(_rvbk4_bytes(_bklen)),
      bitstream_max_len(bitstream_words_for(inlen, _use_HFR)),
      use_HFR(_use_HFR),
      is_comp(_is_comp),
      archive_dst((PHF_BYTE*)_archive_dst)
  {
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    pbkgo_max_blocks_per_sm =
        phf::module::HFR_PBKGO_encode<SYM, 10, 2, uint32_t, 128>::max_blocks_per_sm();
    pbkgo_max_resident_blocks = pbkgo_max_blocks_per_sm * num_sms;

    this->use_sublen_1ki = use_sublen_1ki;
    sublen = tune_sublen(inlen, use_HFR, use_sublen_1ki);
    pardeg = (inlen - 1) / sublen + 1;
    scan_num_tiles_ = tune_scan_tiles(pardeg_cap(inlen, use_HFR));

    h_book4 = MAKE_UNIQUE_HOST(H4, bklen);
    h_hist = MAKE_UNIQUE_HOST(u4, bklen);
    h_rvbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, rvbk4_bytes);
    h_par_entry = MAKE_UNIQUE_HOST(M, pardeg_cap(inlen, use_HFR));
    h_encoded = nullptr;  // no pinned mirror; encoded_h() has no callers

    for (int i = 0; i < 3; ++i) timing_events[i] = _ptb::make_gpu_event();

    if (use_HFR) {
      pbk_book = (H4*)pbk25_r128_book_d_ptr();
      pbk_rvbk = (PHF_BYTE*)pbk25_r128_rvbk_d_ptr();
      pbk_lut = (phf::LutEntry*)pbk25_r128_lut_d_ptr();
    }
  }

  bool set_inlen(size_t inlen, bool _use_sublen_1ki)
  {
    if (inlen > max_inlen) return false;
    if (inlen == len and _use_sublen_1ki == use_sublen_1ki) return true;
    len = inlen;
    use_sublen_1ki = _use_sublen_1ki;
    sublen = tune_sublen(inlen, use_HFR, use_sublen_1ki);
    pardeg = (inlen - 1) / sublen + 1;
    return true;
  }

  static size_t bitstream_words_for(size_t inlen, bool use_HFR)
  {
    if (not use_HFR) return words(sizeof(SYM) * inlen);

    size_t most = 0;
    for (int magnitude : {10, 11, 12}) {
      auto const nblock = (inlen - 1) / ((size_t)1 << magnitude) + 1;
      auto const want = nblock * psz::hfr_stride_words<SYM>(magnitude);
      if (want > most) most = want;
    }
    return most;
  }

  void dump() const
  {
    static char const* nm[NUM_SYM] = {"scratch",     "book",         "rvbk",
                                      "bitstream",   "par_nbit",     "par_ncell",
                                      "par_entry",   "scan_partial", "scan_incl",
                                      "scan_status", "pick_encid",   "incomp_flag",
                                      "pbkgo_state", "lut",          "hist",
                                      "total_ncell", "hf_header",    "info_total_ncell",
                                      "pbk_headers"};
    auto mib = [](size_t b) { return b / 1048576.0; };
    printf(
        "  -- %s %s  E=u%zu len=%zu pardeg=%zu: data %.2f MiB, state %.3f MiB --\n",
        use_HFR ? "HFR" : "HF ", is_comp ? "encode" : "decode", sizeof(SYM), len, pardeg,
        mib(data().bytes()), mib(state().bytes()));
    for (size_t i = 0; i < data().n_frame(); i++)
      if (data().nbyte(i))
        printf(
            "     %-20s %12zu B  %8.2f MiB%s\n", nm[data().sym(i)], data().nbyte(i),
            mib(data().nbyte(i)), data().owns(i) ? "" : "   (alias)");
    for (size_t i = 0; i < state().n_frame(); i++)
      if (state().nbyte(i))
        printf(
            "     %-20s %12zu B  %8.3f MiB\n", nm[state().sym(i)], state().nbyte(i),
            mib(state().nbyte(i)));
  }

  void init() override
  {
    if (getenv("PSZ_DUMP_FRAMES")) dump();

    phf::_dummy::launch();  // call dummy during expensive init

    d_data = MAKE_UNIQUE_DEVICE(u1, data().bytes());
    d_state = MAKE_UNIQUE_DEVICE(u1, state().bytes());
    attach(d_data.get(), d_state.get());
  }

  void init_state() override
  {
    if (scan_partial())
      psz::scan_lookback::launch_init_host(
          scan_partial(), scan_incl(), scan_status(), scan_num_tiles_, /*stream*/ 0);
    cudaDeviceSynchronize();
  }

  void reset(void* stream) override
  {
    if (not is_comp) return;  // a decode buf declares none of the frames below
    if (scan_partial())
      psz::scan_lookback::launch_init_host(
          scan_partial(), scan_incl(), scan_status(), scan_num_tiles_, stream);
    if (use_HFR) {
      if (pbkgo_state())
        cudaMemsetAsync(pbkgo_state(), 0, pardeg * sizeof(u4), (cudaStream_t)stream);
      cudaMemsetAsync(pbk_headers(), 0, pardeg * 2 * sizeof(u4), (cudaStream_t)stream);
    }
  }

  H4* scratch() const { return data().ptr_of<H4>(SCRATCH); }
  H4* book() const { return data().ptr_of<H4>(BOOK); }
  PHF_BYTE* rvbk() const { return data().ptr_of<PHF_BYTE>(RVBK); }
  H4* bitstream() const { return data().ptr_of<H4>(BITSTREAM); }
  M* par_nbit() const { return data().ptr_of<M>(PAR_NBIT); }
  M* par_ncell() const { return data().ptr_of<M>(PAR_NCELL); }
  M* par_entry() const { return data().ptr_of<M>(PAR_ENTRY); }
  PHF_BYTE* archive_bitstream() const
  { return encoded() + PHFHEADER_FORCED_ALIGN + (use_prebuilt_rvbk ? 0 : rvbk4_bytes); }

  PHF_BYTE* encoded() const { return archive_dst ? archive_dst : (PHF_BYTE*)scratch(); }

  u4* scan_partial() const { return state().ptr_of<u4>(SCAN_PARTIAL); }
  u4* scan_incl() const { return state().ptr_of<u4>(SCAN_INCL); }
  int* scan_status() const { return state().ptr_of<int>(SCAN_STATUS); }
  u4* total_ncell() const
  {
    auto const p = state().ptr_of<u4>(TOTAL_NCELL);
    return p ? p : state().ptr_of<u4>(INFO_TOTAL_NCELL);
  }
  u4* pick_encid() const { return state().ptr_of<u4>(PICK_ENCID); }
  u4* hist() const { return state().ptr_of<u4>(HIST); }
  BHeader* pbk_headers() const
  {
    auto const p = state().ptr_of<BHeader>(PBK_HEADERS);
    return p ? p : state().ptr_of<BHeader>(HF_HEADER);
  }
  u1* incomp_flag() const { return state().ptr_of<u1>(INCOMP_FLAG); }
  u4* pbkgo_state() const { return state().ptr_of<u4>(PBKGO_STATE); }
  phf::LutEntry* lut() const { return state().ptr_of<phf::LutEntry>(LUT); }

  // public functions
  void memcpy_merge(Header& header, phf_stream_t stream)

  {
    auto memcpy_start = encoded();
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _rvbk{rvbk(), rvbk4_bytes, header.entry[PHFHEADER_RVBK]};
    // SoA metadata retired: the per-block header ships as bheader AoS.
    memcpy_helper _par_nbit{par_nbit(), size_t{0}, header.entry[PHFHEADER_PAR_NBIT]};
    memcpy_helper _par_entry{par_entry(), size_t{0}, header.entry[PHFHEADER_PAR_ENTRY]};
    H4* bitstream_src = (use_HFR and not use_pbkgo) ? nullptr : bitstream();
    memcpy_helper _bitstream{
        bitstream_src, bitstream_src ? (size_t)header.total_ncell * sizeof(H4) : 0,
        header.entry[PHFHEADER_BITSTREAM]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
      if (var.nbyte == 0) return;  // skip dead sections (PBKC's RVBK, brnum=0 sections)
      cudaMemcpyAsync(
          start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    };

    static_assert(sizeof(Header) <= PHFHEADER_FORCED_ALIGN, "phf_header exceeds its aligned slot");
    PHF_BYTE header_padded[PHFHEADER_FORCED_ALIGN] = {};
    memcpy(header_padded, &header, sizeof(header));
    cudaMemcpyAsync(
        start, header_padded, PHFHEADER_FORCED_ALIGN, cudaMemcpyHostToDevice,
        (cudaStream_t)stream);

    if (use_global_encid)  // HFR-v3
      cudaMemcpyAsync(
          start + offsetof(Header, g_encid), pick_encid(), sizeof(u1), cudaMemcpyDeviceToDevice,
          (cudaStream_t)stream);

    if (not use_prebuilt_rvbk) d2d_memcpy_merge(_rvbk);  // not applicable for PBK
    d2d_memcpy_merge(_par_nbit);
    d2d_memcpy_merge(_par_entry);
    d2d_memcpy_merge(_bitstream);

    // header.pardeg, not the buf's: 2Ki+ PBKC encodes fewer blocks than the 1Ki default.
    auto const seg =
        state().find(PBK_HEADERS) != npos ? PHFHEADER_PBK_HEADERS : PHFHEADER_HF_REV2_HEADER;
    memcpy_helper _headers{
        (u4*)pbk_headers(), 2 * (size_t)header.pardeg * sizeof(u4), header.entry[seg]};
    d2d_memcpy_merge(_headers);
  }

  [[deprecated]] void clear_buffer()
  {
    memset_device(scratch(), len);
    memset_device(book(), bklen);
    memset_device(rvbk(), rvbk4_bytes);
    memset_device(bitstream(), bitstream_max_len);
    memset_device(par_nbit(), pardeg);
    memset_device(par_ncell(), pardeg);
    memset_device(par_entry(), pardeg);
  }
};

#define PHF_BUF_DEF(...) \
  template <typename E>  \
  __VA_ARGS__ phf::Buf<E>

PHF_BUF_DEF()::Buf(size_t inlen, size_t bklen, bool use_sublen_1ki, bool is_comp) :
    Buf(inlen, bklen, use_sublen_1ki, is_comp, /*use_HFR=*/false, nullptr)
{
}

PHF_BUF_DEF()::Buf(
    size_t inlen, size_t bklen, bool use_sublen_1ki, bool is_comp, bool use_HFR, void* archive_dst)
{
  auto t = use_HFR ? impl::plan_HFR(inlen, bklen, is_comp, archive_dst != nullptr)
                   : impl::plan_HF(inlen, bklen, is_comp);

  pimpl = std::make_unique<impl>(
      inlen, bklen, use_HFR, use_sublen_1ki, is_comp, archive_dst, std::move(t));
}

PHF_BUF_DEF()::~Buf() {}

template <typename E>
phf::Buf_HFR<E>::Buf_HFR(
    size_t inlen, size_t bklen, bool use_sublen_1ki, bool is_comp, void* archive_dst) :
    Buf<E>(inlen, bklen, use_sublen_1ki, is_comp, /*use_HFR=*/true, archive_dst)
{
}

template <typename E>
phf::Buf_HFR<E>::~Buf_HFR() = default;

PHF_BUF_DEF(void)::init() { pimpl->init(); }
PHF_BUF_DEF(size_t)::planned_data_bytes() const { return pimpl->data().bytes(); }
PHF_BUF_DEF(size_t)::planned_state_bytes() const { return pimpl->state().bytes(); }
PHF_BUF_DEF(void)::attach(void* d_data, void* d_state) { pimpl->attach(d_data, d_state); }

// a series of getters: variables
PHF_BUF_DEF(size_t)::rvbk_bytes() const { return pimpl->rvbk4_bytes; }
PHF_BUF_DEF(u2)::rt_bklen() const { return pimpl->rt_bklen; }
PHF_BUF_DEF(int)::num_sms() const { return pimpl->num_sms; }
PHF_BUF_DEF(int)::pbkgo_max_blocks_per_sm() const { return pimpl->pbkgo_max_blocks_per_sm; }
PHF_BUF_DEF(int)::pbkgo_max_resident_blocks() const { return pimpl->pbkgo_max_resident_blocks; }
PHF_BUF_DEF(bool)::set_inlen(size_t inlen, bool use_sublen_1ki)
{ return pimpl->set_inlen(inlen, use_sublen_1ki); }
PHF_BUF_DEF(size_t)::sublen() const { return pimpl->sublen; }
PHF_BUF_DEF(size_t)::pardeg() const { return pimpl->pardeg; }
PHF_BUF_DEF(size_t)::bitstream_max_len() const { return pimpl->bitstream_max_len; }
PHF_BUF_DEF(size_t)::archive_max_words(size_t inlen, size_t bklen, bool use_HFR)
{ return impl::archive_max_words(inlen, impl::_rvbk4_bytes((int)bklen), use_HFR); }

// a series of getters: arrays
PHF_BUF_DEF(H4*)::book_d() const { return pimpl->book(); }
PHF_BUF_DEF(H4*)::book_h() const { return pimpl->h_book4.get(); }
PHF_BUF_DEF(u1*)::rvbk_d() const { return pimpl->rvbk(); }
PHF_BUF_DEF(u1*)::rvbk_h() const { return pimpl->h_rvbk4.get(); }
PHF_BUF_DEF(H4*)::scratch_d() const { return pimpl->scratch(); }
PHF_BUF_DEF(H4*)::scratch_h() const { return pimpl->h_scratch4.get(); }
PHF_BUF_DEF(M*)::par_nbit_d() const { return pimpl->par_nbit(); }
PHF_BUF_DEF(M*)::par_nbit_h() const { return pimpl->h_par_nbit.get(); }
PHF_BUF_DEF(M*)::par_ncell_d() const { return pimpl->par_ncell(); }
PHF_BUF_DEF(M*)::par_ncell_h() const { return pimpl->h_par_ncell.get(); }
PHF_BUF_DEF(M*)::par_entry_d() const { return pimpl->par_entry(); }
PHF_BUF_DEF(M*)::par_entry_h() const { return pimpl->h_par_entry.get(); }
PHF_BUF_DEF(H4*)::bitstream_d() const { return pimpl->bitstream(); }
PHF_BUF_DEF(H4*)::bitstream_h() const { return pimpl->h_bitstream4.get(); }
PHF_BUF_DEF(PHF_BYTE*)::encoded_d() const { return pimpl->encoded(); }
PHF_BUF_DEF(PHF_BYTE*)::encoded_h() const { return pimpl->h_encoded; }

PHF_BUF_DEF(u4*)::scan_partial_aggregate_d() const { return pimpl->scan_partial(); }
PHF_BUF_DEF(u4*)::scan_incl_prefix_d() const { return pimpl->scan_incl(); }
PHF_BUF_DEF(int*)::scan_tile_status_d() const { return pimpl->scan_status(); }
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
  auto const hdr_bytes = 2 * header.pardeg * sizeof(u4);
  auto const declares = [this](int sym) {
    return pimpl->state().find(sym) != _ptb::mem_plan::npos;
  };
  byte_offsets[PHFHEADER_PBK_HEADERS] = declares(impl::PBK_HEADERS) ? hdr_bytes : 0;
  byte_offsets[PHFHEADER_HF_REV2_HEADER] = declares(impl::HF_HEADER) ? hdr_bytes : 0;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = byte_offsets[i - 1];
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
}

PHF_BUF_DEF(void)::set_use_prebuilt_rvbk(bool v) { pimpl->use_prebuilt_rvbk = v; }
PHF_BUF_DEF(void)::set_use_pbkgo(bool v) { pimpl->use_pbkgo = v; }
PHF_BUF_DEF(void)::set_use_global_encid(bool v) { pimpl->use_global_encid = v; }
PHF_BUF_DEF(u4*)::pick_encid_d() const { return pimpl->pick_encid(); }
PHF_BUF_DEF(u4*)::hist_d() const { return pimpl->hist(); }
PHF_BUF_DEF(u4*)::hist_h() const { return pimpl->h_hist.get(); }
PHF_BUF_DEF(void*)::timing_event(int idx) const { return (void*)pimpl->timing_events[idx].get(); }

// method: set internal variable
PHF_BUF_DEF(void)::set_rt_bklen(const int _rt_bklen) { pimpl->rt_bklen = _rt_bklen; }

PHF_BUF_DEF(void)::memcpy_merge(phf_header& header, phf_stream_t stream)
{ pimpl->memcpy_merge(header, stream); }

// method, same-name
PHF_BUF_DEF(void)::clear_buffer() { pimpl->clear_buffer(); }  // NOLINT

PHF_BUF_DEF(void)::reset(phf_stream_t stream) { pimpl->reset(stream); }
PHF_BUF_DEF(void)::reset_HFR(phf_stream_t stream) { pimpl->reset(stream); }

PHF_BUF_DEF(psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>*)::pbk_headers_d() const
{ return pimpl->pbk_headers(); }
PHF_BUF_DEF(psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>*)::pbk_headers_h() const
{ return pimpl->h_pbk_headers.get(); }
PHF_BUF_DEF(u1*)::archive_bitstream_d() const { return pimpl->archive_bitstream(); }
PHF_BUF_DEF(u4*)::total_ncell_d() const { return pimpl->total_ncell(); }
PHF_BUF_DEF(u1*)::incomp_flag_d() const { return pimpl->incomp_flag(); }
PHF_BUF_DEF(u4*)::pbkgo_state_d() const { return pimpl->pbkgo_state(); }
PHF_BUF_DEF(phf::LutEntry*)::lut_d() const { return pimpl->lut(); }
PHF_BUF_DEF(H4*)::pbk_book_d() const { return pimpl->pbk_book; }
PHF_BUF_DEF(u1*)::pbk_rvbk_d() const { return pimpl->pbk_rvbk; }
PHF_BUF_DEF(phf::LutEntry*)::pbk_lut_d() const { return pimpl->pbk_lut; }

}  // namespace phf

template struct phf::Buf<u1>;
template struct phf::Buf<u2>;
template struct phf::Buf<u4>;
template struct phf::Buf_HFR<u1>;
template struct phf::Buf_HFR<u2>;
template struct phf::Buf_HFR<u4>;

#undef PHF_BUF_DEF