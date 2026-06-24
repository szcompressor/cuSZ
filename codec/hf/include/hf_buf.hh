#ifndef HF_BUF_HH_PRIVATE
#define HF_BUF_HH_PRIVATE

#include <cstddef>
#include <memory>

#include "c_type.h"
#include "hf.h"
#include "hfr-pbk.hh"

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
  using BlockOutlierCell = _ptb::compact_cell<f4, u2>;

  // ctor/dtor
  Buf(size_t inlen, size_t _bklen, int _pardeg = -1, bool _use_HFR = false, bool debug = false,
      bool use_sublen_1ki = false);
  ~Buf();

  // utils
  void set_rt_bklen(int const rt_bklen);
  auto rt_bklen() const -> u2;
  auto num_sms() const -> int;
  auto sublen() const -> size_t;
  auto pardeg() const -> size_t;
  auto bitstream_max_len() const -> size_t;
  auto rvbk_bytes() const -> size_t;
  auto set_use_prebuilt_rvbk(bool v) -> void;
  auto set_use_pbkgo(bool v) -> void;
  auto set_use_global_encid(bool v) -> void;  // HFR-v3 uses global PBK ID, async cp'ed to header
  auto pick_encid_d() const -> u4*;
  auto timing_event(int idx) const -> void*;    // 3 reusable cudaEvent_t vars
  auto pbkgo_max_blocks_per_sm() const -> int;  // PBKGO: computed at init, 0 for sizeof(SYM) > 2.
  auto pbkgo_max_resident_blocks() const -> int;

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

  // scan state for PBKGO
  u4* scan_partial_aggregate_d() const;
  u4* scan_incl_prefix_d() const;
  int* scan_tile_status_d() const;
  int scan_num_tiles() const;

  // HFR-PBK family (allocated on use_HFR=true).
  BHeader* pbk_headers_d() const;
  BHeader* pbk_headers_h() const;
  H4* packed_d() const;
  u4* total_ncell_d() const;
  u4* pbk_packed_headers_d() const;  // 2 u4 per block; HFR family only
  BlockOutlierCell* block_outliers_d() const;
  u1* incomp_flag_d() const;
  u4* pbkgo_state_d() const;

  void update_header(phf_header& header);
  void calc_offset(phf_header& header, M* byte_offsets);

  // misc. methods
  void memcpy_merge(phf_header& header, phf_stream_t stream);
  void clear_buffer();
  void reset(phf_stream_t stream);      // per-encode clear of scan state
  void reset_HFR(phf_stream_t stream);  // per-encode clear ofor HFR*
};

}  // namespace phf

#endif /* HF_BUF_HH_PRIVATE */
