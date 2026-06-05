// PRIVATE — full definition of phf::Buf buffer management class.

#ifndef HF_BUF_HH_PRIVATE
#define HF_BUF_HH_PRIVATE

#include <memory>

#include "hfr-pbk.hh"  // psz::HFR_PBK_Breaks<128>
#include "c_type.h"
#include "hf.h"

// fwd-decl, full definition lives in codec/hf/include/hfr-pbk.hh
namespace psz {
template <u2 Radius>
struct HFR_PBK_Breaks;
}

namespace phf {

template <typename E>
struct Buf {
  struct impl;
  std::unique_ptr<impl> pimpl;

  // helper
  typedef struct RC {
    static const int SCRATCH = 0;
    static const int FREQ = 1;
    static const int BK = 2;
    static const int RVBK = 3;
    static const int PAR_NBIT = 4;
    static const int PAR_NCELL = 5;
    static const int PAR_ENTRY = 6;
    static const int BITSTREAM = 7;
    static const int END = 8;
  } RC;

  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  static constexpr auto Radius = psz::HFR_PBK_Constants::Radius;

  using SYM = E;
  using H4 = u4;
  using M = PHF_METADATA;
  using Header = phf_header;
  using BHeader = psz::_future::bheader<E, Radius>;

  // constructor/destructor
  Buf(size_t inlen, size_t _bklen, int _pardeg = -1, bool _use_HFR = false, bool debug = false);
  ~Buf();

  // setter
  void register_runtime_bklen(int const rt_bklen);

  // getter: variables
  u2 rt_bklen() const;
  int numSMs() const;
  size_t sublen() const;
  size_t pardeg() const;
  size_t bitstream_max_len() const;
  size_t rvbk_bytes() const;
  // True when the runtime rvbk needn't ship (PBKC uses the baked-in pbk25_r128).
  void set_omit_runtime_rvbk(bool v);
  void set_use_pbkgo(bool v);

  // Reusable cudaEvent_t (idx ∈ {0,1,2}); void* keeps this header CUDA-free.
  void* timing_event(int idx) const;
  // Encoder sets this so HF_rev2's AoS bheader_backport[] section goes live.
  void set_use_hf_rev2_header(bool v);

  // HFR-PBK-GO launch budget; computed at init, 0 for sizeof(SYM) > 2.
  int pbkgo_max_blocks_per_sm() const;
  int pbkgo_max_resident_blocks() const;

  // getter: arrays
  H4* book_d() const;
  H4* book_h() const;
  u1* rvbk_d() const;
  u1* rvbk_h() const;
  H4* scratch_d() const;
  H4* scratch_h() const;
  M* par_nbit_d() const;
  M* par_nbit_h() const;
  M* par_ncell_d() const;
  M* par_ncell_h() const;
  M* par_entry_d() const;
  M* par_entry_h() const;
  H4* bitstream_d() const;
  H4* bitstream_h() const;
  PHF_BYTE* encoded_d() const;
  PHF_BYTE* encoded_h() const;

  psz::HFR_PBK_Breaks<128>* sp_breaks_d() const;
  u4* sp_count_d() const;
  u4* par_brnum_d() const;
  u4* par_brnum_h() const;
  u4* par_broffset_d() const;
  u4* par_broffset_h() const;
  // Per-block enc_id: 0 = normal Huffman, 1 = incomp (raw symbols inline).
  u1* par_encid_d() const;
  u1* par_encid_h() const;

  // Decoupled-lookback scan state for LAGO concat (psz::scan_lookback).
  u4* scan_partial_aggregate_d() const;
  u4* scan_incl_prefix_d() const;
  int* scan_tile_status_d() const;
  int scan_num_tiles() const;

  // HFR-PBK family scratch (allocated only when use_HFR=true; nullptr otherwise).
  BHeader* pbk_headers_d() const;
  BHeader* pbk_headers_h() const;
  H4* packed_d() const;              // post-concat compact bitstream, pardeg * BlockSize words
  u4* total_ncell_d() const;         // 1-word; total compact size from LAGO concat
  u4* total_nbit_d() const;          // 1-word; reduce_total_nbit sink
  u4* pbk_packed_headers_d() const;  // 2 u4 per block; HFR family only
  u4* pbkgo_state_d() const;  // PBKGO decoupled-lookback state, 1 u4 / block
  u4* hf_rev2_header_d() const;  // bheader_backport[] = 2 u4 per block; HF_rev2 only

  void update_header(phf_header& header);
  void calc_offset(phf_header& header, M* byte_offsets);

  // other methods
  void memcpy_merge(phf_header& header, phf_stream_t stream);
  void clear_buffer();
  // Per-encode reset: scan state.
  void reset(phf_stream_t stream);
  // Per-encode reset for the HFR family.
  void reset_HFR(phf_stream_t stream);
};

}  // namespace phf

#endif /* HF_BUF_HH_PRIVATE */
