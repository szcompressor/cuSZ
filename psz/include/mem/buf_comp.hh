#ifndef PSZ_COMPBUF_HH
#define PSZ_COMPBUF_HH

#include <../../codec/hf/include/hf_buf.hh>  // needed for Buf instantiation
#include <cstdint>
#include <memory>

#include "cusz/type.h"
#include "hf_hl.hh"
#include "lc_gen/lc_buf.h"
#include "mem/cxx_sp_gpu.h"

// segment
// base
// #define PSZ_HEADER 0
// #define PSZ_ANCHOR 1
// #define PSZ_ENCODED 2
// #define PSZ_SPFMT 3
// #define PSZ_END 4

// incoming: spline
#define PSZ_HEADER 0
#define PSZ_ENCODED 1
#define PSZ_ANCHOR 2
#define PSZ_SPFMT 3
// #define PSZ_END 4
#define PSZ_ENC_PASS1_END 4
#define PSZ_ENC_PASS2_END 5

namespace psz {

// namespace-wide type aliases
using M = u4;
using Freq = u4;
using BYTE = u1;
using H = u4;

struct BufToggle_Comp {
  bool use_quant;
  bool use_outlier;
  bool use_anchor;
  bool use_hist;
  bool use_compressed;
  bool use_top1;
};

template <typename _T, typename _E = u2>
struct Buf_Comp {
 public:
  using T = _T;
  using E = _E;
  using FP = T;
  using M = uint32_t;

  using Buf_Outlier = _ptb::compact_gpu<T>;
  using Buf_Outlier2 = _ptb::compact_GPU_DRAM2<T, M>;
  using Buf_HF = phf::Buf<E>;
  using Buf_LC = LC_Buf;

  struct impl;
  std::unique_ptr<impl> pimpl;

  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;
  constexpr static float OUTLIER_RATIO = 0.1;

  // spline-specific anchor spacing .
  constexpr static int BLK16 = 16;  // y25 (2D+3D)
  constexpr static int BLK8 = 8;    // y24 (lean 3D)
  constexpr static int ERR_HISTO_LEN = 36;

  // selector: (0 = y25/BLK16, 1 = y24/BLK8); does not change Buf_Comp ABI
  void set_spline_variant(int v);

  bool is_comp;
  // const u4 x, y, z;
  const psz_len len;
  const size_t len_linear;

  // encapsulations
  int hist_generic_grid_dim;
  int hist_generic_block_dim;
  int hist_generic_shmem_use;
  int hist_generic_repeat;
  BYTE* comp_codec_out{nullptr};
  size_t comp_codec_outlen{0};
  uint32_t nbyte[PSZ_ENC_PASS2_END];

  psz_header* header_ref;

 public:
  Buf_Comp(psz_len len, BufToggle_Comp* toggle);
  Buf_Comp(
      psz_len len, bool _is_comp = true, bool use_HFR = false, bool alloc_eq = true,
      bool use_sublen_1ki = false, bool tile_order = false, bool y25_tile = false);
  ~Buf_Comp();

  void register_header(psz_header* header) { header_ref = header; }

  void clear_buffer();
  void clear_top1();

  // getter
  E* eq_d() const;
  psz_len eq_len3() const;
  T* decode_fused_d() const;
  size_t eq_len() const;
  void alloc_decode_fused();
  OutlierCell* block_outliers_d() const;

  Freq* hist_d() const;
  Freq* hist_h() const;
  Freq* top1_d() const;
  Freq* top1_h() const;
  size_t top1_nblk() const;

  T* anchor_d() const;
  size_t anchor_len() const;
  psz_len anchor_len3() const;

  BYTE* compressed_d() const;
  BYTE* compressed_h() const;

  Buf_Outlier2* buf_outlier2() const;
  void* outlier2_validx_d() const;
  M outlier2_host_get_num() const;
  size_t outlier2_max_allowed_num() const;

  // extra for profifling
  T* profiled_errors_d() const;
  T* profiled_errors_h() const;
  M profiled_errors_len() const;

  Buf_HF* buf_hf() const;
  Buf_LC* buf_lc() const;

  float outlier_ratio() const { return OUTLIER_RATIO; };
};

}  // namespace psz

#endif /* PSZ_COMPBUF_HH */
