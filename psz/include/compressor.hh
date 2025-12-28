#ifndef PSZ_COMPRESSOR_HH
#define PSZ_COMPRESSOR_HH

#include "cusz/context.h"
#include "cusz/header.h"
#include "cusz/type.h"
#include "hf_hl.hh"
#include "mem/buf_comp.hh"

namespace psz {

template <typename T>
class [[deprecated("use non-OOD compression pieline instead")]] Compressor {
 private:
 public:
  using E = u2;
  using M = u4;
  using BYTE = u1;
  using H = u4;
  using Buf = Buf_Comp<T>;

  // encapsulations
  int hist_generic_grid_dim;
  int hist_generic_block_dim;
  int hist_generic_shmem_use;
  int hist_generic_repeat;
  size_t len_linear;
  BYTE* comp_codec_out{nullptr};
  size_t comp_codec_outlen{0};
  uint32_t nbyte[PSZ_ENC_PASS2_END];
  float time_sp;
  double eb, eb_r, ebx2, ebx2_r;
  psz_header* const header_ref;

  Buf* mem;
  phf::Buf<E>* buf_hf;

 private:
  void compress_predict_enc1(psz_ctx* ctx, T* in, void* stream);
  void compress_enc1_wrapup(psz_ctx* ctx, BYTE** out, size_t* outlen, void* stream);

 public:
  // comp, ctor(...) + no further init
  Compressor(psz_ctx*);
  Compressor(psz_header*);
  // dtor, releasing
  ~Compressor();

  void compress(psz_ctx*, T*, BYTE**, size_t*, psz_stream_t);
  void decompress(psz_header*, BYTE*, T*, psz_stream_t);
  void dump_compress_intermediate(psz_ctx*, psz_stream_t);
  void clear_buffer();

  // getter
  void export_header(psz_header&);
};

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

using CompressorF4 = Compressor<f4>;
using CompressorF8 = Compressor<f8>;

}  // namespace psz

template <typename T, typename E>
using psz_buf = psz::Buf_Comp<T, E>;

#define PSZ_BUF psz_buf<T, E>

namespace psz {

template <typename T, typename E>
struct compression_pipeline {
  static void* compress_init(psz_ctx* ctx);
  static void* decompress_init(psz_header* header);
  static int compress(psz_ctx*, PSZ_BUF* mem, T*, u1**, size_t*, psz_stream_t);
  static int compress_analysis(psz_ctx*, PSZ_BUF* mem, T*, u4*, psz_stream_t);
  static int decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream);
  static void release(PSZ_BUF* mem);
  static void compress_dump_internal_buf(psz_ctx* ctx, PSZ_BUF* mem, psz_stream_t stream);
};

}  // namespace psz

#endif /* PSZ_COMPRESSOR_HH */
