#include <array>
#include <cstdlib>

#include "hf.h"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "utils/io.hh"

using stdlen3 = std::array<size_t, 3>;
using _portable::utils::fromfile;
using _portable::utils::tofile;

namespace psz {

struct CompressorBufferToggle {
  bool err_ctrl_quant{true};
  bool compact_outlier{true};
  bool anchor{true};
  bool histogram{false};
  bool compressed{false};
  bool top1{false};
  bool pbk_all{false};
};

template <typename DType>
class CompressorBuffer {
 public:
  using T = DType;
  using E = u2;
  using FP = T;
  using M = u4;
  using Freq = u4;
  using B = u1;
  using Compact = _portable::compact_gpu<T>;
  using Hf = u4;

  GPU_unique_dptr<E[]> d_ectrl;
  GPU_unique_dptr<T[]> d_anchor;
  GPU_unique_dptr<B[]> d_compressed;
  GPU_unique_hptr<B[]> h_compressed;
  GPU_unique_dptr<Freq[]> d_hist;
  GPU_unique_dptr<Freq[]> d_top1;
  GPU_unique_hptr<Freq[]> h_top1;

  static constexpr auto ChunkSize = 1024;
  static constexpr auto PBK_LEN = 128;
  static constexpr auto PBK_N = 11;
  u4 PBK_REVBK_BYTES;
  GPU_unique_dptr<Hf[]> d_pbk_hist;
  GPU_unique_hptr<Hf[]> h_pbk_hist;
  GPU_unique_dptr<Hf[]> d_pbk_r64;
  GPU_unique_hptr<Hf[]> h_pbk_r64;
  GPU_unique_dptr<u1[]> d_pbk_revbk_r64;
  GPU_unique_hptr<u1[]> h_pbk_revbk_r64;
  GPU_unique_dptr<u4[]> d_pbk_bitstream;
  GPU_unique_hptr<u4[]> h_pbk_bitstream;
  GPU_unique_dptr<u2[]> d_pbk_bits;
  GPU_unique_hptr<u2[]> h_pbk_bits;
  GPU_unique_dptr<u4[]> d_pbk_entries;
  GPU_unique_hptr<u4[]> h_pbk_entries;
  GPU_unique_dptr<u1[]> d_pbk_tree_IDs;
  GPU_unique_hptr<u1[]> h_pbk_tree_IDs;
  GPU_unique_dptr<size_t[]> d_pbk_loc;
  GPU_unique_hptr<size_t[]> h_pbk_loc;
  // breaking handling
  GPU_unique_dptr<E[]> d_pbk_brval;
  GPU_unique_dptr<M[]> d_pbk_bridx;
  GPU_unique_dptr<M[]> d_pbk_brnum;
  GPU_unique_hptr<M[]> h_pbk_brnum;

  const char* shellvar_pbk_hist;
  const char* shellvar_pbk_book;
  const char* shellvar_pbk_rvbk;
  u4 num_chunk;
  size_t pbk_bytes;

  using Toggle = CompressorBufferToggle;

  Compact* compact;
  bool const is_comp;

  constexpr static size_t BLK = 8;  // for spline
  constexpr static u2 max_radius = 512;
  constexpr static u2 max_bklen = max_radius * 2;

  u4 const x, y, z;
  size_t const len;
  size_t const anchor512_len;  // for spline

 private:
  static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

  static size_t set_len_anchor_512(u4 x, u4 y, u4 z)
  {
    return _div(x, BLK) * _div(y, BLK) * _div(z, BLK);
  }

  void init_with_toggles(Toggle* toggle)
  {
    if (toggle->err_ctrl_quant) d_ectrl = MAKE_UNIQUE_DEVICE(E, len);
    if (toggle->compact_outlier) compact = new Compact(len / 5);
    if (toggle->anchor) d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
    if (toggle->histogram) d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
    if (toggle->compressed) {
      d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
    }
    if (toggle->top1) {
      d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
      h_top1 = MAKE_UNIQUE_HOST(Freq, 1);
    }

    if (toggle->pbk_all) {
      shellvar_pbk_book = std::getenv("PBK_BOOK");
      shellvar_pbk_rvbk = std::getenv("PBK_RVBK");

      num_chunk = (len + ChunkSize - 1) / ChunkSize;

      d_pbk_r64 = MAKE_UNIQUE_DEVICE(Hf, PBK_LEN * PBK_N);
      h_pbk_r64 = MAKE_UNIQUE_HOST(Hf, PBK_LEN * PBK_N);
      d_pbk_bitstream = MAKE_UNIQUE_DEVICE(u4, len / 2);
      h_pbk_bitstream = MAKE_UNIQUE_HOST(u4, len / 2);
      d_pbk_bits = MAKE_UNIQUE_DEVICE(u2, num_chunk);
      h_pbk_bits = MAKE_UNIQUE_HOST(u2, num_chunk);
      d_pbk_entries = MAKE_UNIQUE_DEVICE(u4, num_chunk);
      h_pbk_entries = MAKE_UNIQUE_HOST(u4, num_chunk);
      d_pbk_tree_IDs = MAKE_UNIQUE_DEVICE(u1, num_chunk);
      h_pbk_tree_IDs = MAKE_UNIQUE_HOST(u1, num_chunk);
      d_pbk_loc = MAKE_UNIQUE_DEVICE(size_t, num_chunk);
      h_pbk_loc = MAKE_UNIQUE_HOST(size_t, num_chunk);

      d_pbk_brval = MAKE_UNIQUE_DEVICE(E, 100 + num_chunk);
      d_pbk_bridx = MAKE_UNIQUE_DEVICE(M, 100 + num_chunk);
      d_pbk_brnum = MAKE_UNIQUE_DEVICE(M, 1);
      h_pbk_brnum = MAKE_UNIQUE_HOST(M, 1);

      fromfile(shellvar_pbk_book, h_pbk_r64.get(), PBK_LEN * PBK_N);
      memcpy_allkinds<H2D>(d_pbk_r64.get(), h_pbk_r64.get(), PBK_LEN * PBK_N);

      PBK_REVBK_BYTES = phf_reverse_book_bytes(PBK_LEN, 4, sizeof(E));
      d_pbk_revbk_r64 = MAKE_UNIQUE_DEVICE(u1, PBK_REVBK_BYTES * PBK_N);
      h_pbk_revbk_r64 = MAKE_UNIQUE_HOST(u1, PBK_REVBK_BYTES * PBK_N);

      fromfile(shellvar_pbk_rvbk, h_pbk_revbk_r64.get(), PBK_REVBK_BYTES * PBK_N);
      memcpy_allkinds<H2D>(d_pbk_revbk_r64.get(), h_pbk_revbk_r64.get(), PBK_REVBK_BYTES * PBK_N);
    }
  }

  void init_compression_defafult()
  {
    d_ectrl = MAKE_UNIQUE_DEVICE(E, ALIGN_4Ki(len));  // align at 4Ki

    compact = new Compact(len / 5);

    d_anchor = MAKE_UNIQUE_DEVICE(T, anchor512_len);
    d_hist = MAKE_UNIQUE_DEVICE(Freq, max_bklen);
    d_compressed = MAKE_UNIQUE_DEVICE(B, len * 4 / 2);
    h_compressed = MAKE_UNIQUE_HOST(B, len * 4 / 2);
    d_top1 = MAKE_UNIQUE_DEVICE(Freq, 1);
    h_top1 = MAKE_UNIQUE_HOST(Freq, 1);
  }

  void init_decompression_default()
  {
    d_ectrl = MAKE_UNIQUE_DEVICE(E, ALIGN_4Ki(len));  // align at 4Ki
  }

  void init_compression_special_singleton()
  {
    compact = new Compact(len / 5);

    num_chunk = (len + ChunkSize - 1) / ChunkSize;

    d_pbk_r64 = MAKE_UNIQUE_DEVICE(Hf, PBK_LEN * PBK_N);
    h_pbk_r64 = MAKE_UNIQUE_HOST(Hf, PBK_LEN * PBK_N);
    d_pbk_bitstream = MAKE_UNIQUE_DEVICE(u4, len / 2);
    h_pbk_bitstream = MAKE_UNIQUE_HOST(u4, len / 2);
    d_pbk_bits = MAKE_UNIQUE_DEVICE(u2, num_chunk);
    h_pbk_bits = MAKE_UNIQUE_HOST(u2, num_chunk);
    d_pbk_entries = MAKE_UNIQUE_DEVICE(u4, num_chunk);
    h_pbk_entries = MAKE_UNIQUE_HOST(u4, num_chunk);
    d_pbk_tree_IDs = MAKE_UNIQUE_DEVICE(u1, num_chunk);
    h_pbk_tree_IDs = MAKE_UNIQUE_HOST(u1, num_chunk);
    d_pbk_loc = MAKE_UNIQUE_DEVICE(size_t, num_chunk);
    h_pbk_loc = MAKE_UNIQUE_HOST(size_t, num_chunk);

    fromfile(shellvar_pbk_book, h_pbk_r64.get(), PBK_LEN * PBK_N);
    memcpy_allkinds<H2D>(d_pbk_r64.get(), h_pbk_r64.get(), PBK_LEN * PBK_N);
  }

  void init_decompression_special_singleton()
  {
    PBK_REVBK_BYTES = phf_reverse_book_bytes(PBK_LEN, 4, sizeof(E));
    d_pbk_revbk_r64 = MAKE_UNIQUE_DEVICE(u1, PBK_REVBK_BYTES * PBK_N);
    h_pbk_revbk_r64 = MAKE_UNIQUE_HOST(u1, PBK_REVBK_BYTES * PBK_N);

    fromfile(shellvar_pbk_rvbk, h_pbk_revbk_r64.get(), PBK_REVBK_BYTES * PBK_N);
    memcpy_allkinds<H2D>(d_pbk_revbk_r64.get(), h_pbk_revbk_r64.get(), PBK_REVBK_BYTES * PBK_N);
  }

 public:
#define TRY_COMPRESS_TIME_USE_PBK if (shellvar_pbk_book)
#define TRY_DECOMPRESS_TIME_USE_PBK if (shellvar_pbk_rvbk)

  CompressorBuffer(u4 x, u4 y = 1, u4 z = 1, bool _is_comp = true, Toggle* toggle = nullptr) :
      is_comp(_is_comp),
      x(x),
      y(y),
      z(z),
      len(x * y * z),
      anchor512_len(set_len_anchor_512(x, y, z))
  {
    if (not toggle) {
      if (is_comp) {
        shellvar_pbk_book = std::getenv("PBK_BOOK");
        TRY_COMPRESS_TIME_USE_PBK
        {
          std::cout << "PBK_BOOK: " << shellvar_pbk_book << std::endl;
          init_compression_special_singleton();
        }
        else
        {
          std::cout << "ENV VAR PBK_BOOK is not set. ";
          std::cout << "fallback to default." << std::endl;
          init_compression_defafult();
        }
      }
      else {
        shellvar_pbk_rvbk = std::getenv("PBK_RVBK");
        TRY_DECOMPRESS_TIME_USE_PBK
        {
          std::cout << "loading PBK_RVBK: " << shellvar_pbk_rvbk << std::endl;
          init_decompression_special_singleton();
        }
        else
        {
          std::cout << "ENV VAR PBK_RVBK is not set. ";
          std::cout << "fallback to default." << std::endl;
          init_decompression_default();
        }
      }
    }
    else {
      init_with_toggles(toggle);
    }
  }

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

  T* anchor() const { return d_anchor.get(); }
  size_t anchor_len() const { return anchor512_len; }
  stdlen3 anchor_len3() const { return stdlen3{_div(x, BLK), _div(y, BLK), _div(z, BLK)}; }

  B* compressed() const { return d_compressed.get(); }
  B* compressed_h() const { return d_compressed.get(); }

  T* compact_val() const { return compact->val(); }
  M* compact_idx() const { return compact->idx(); }
  M compact_num_outliers() const { return compact->num_outliers(); }
  Compact* outlier() { return compact; }

  bool compress_time_use_pbk() const { return shellvar_pbk_book != nullptr; }
  bool decompress_time_use_pbk() const { return shellvar_pbk_rvbk != nullptr; }
  Hf* pbk() const { return d_pbk_r64.get(); }
  Hf* pbk_bitstream() const { return d_pbk_bitstream.get(); }
  Hf* pbk_bitstream_h() const { return h_pbk_bitstream.get(); }
  u2* pbk_bits() const { return d_pbk_bits.get(); }
  u2* pbk_bits_h() const { return h_pbk_bits.get(); }
  u4* pbk_entries() const { return d_pbk_entries.get(); }
  u4* pbk_entries_h() const { return h_pbk_entries.get(); }
  u1* pbk_tree_IDs() const { return d_pbk_tree_IDs.get(); }
  u1* pbk_tree_IDs_h() const { return h_pbk_tree_IDs.get(); }
  size_t* pbk_loc() const { return d_pbk_loc.get(); }

  u1* pbk_revbooks_11() const { return d_pbk_revbk_r64.get(); }
  u1* pbk_revbooks_11_h() const { return h_pbk_revbk_r64.get(); }

  size_t pbk_encoding_endloc() const { return h_pbk_loc[0]; }

  void pbk_encoding_summary(bool print = true)
  {
    memcpy_allkinds<D2H>(h_pbk_loc.get(), d_pbk_loc.get(), 1);
    memcpy_allkinds<D2H>(h_pbk_brnum.get(), d_pbk_brnum.get(), 1);

    size_t bytes_bitstream = h_pbk_loc[0] * sizeof(u4);
    size_t bytes_tree_IDs = num_chunk * sizeof(u1);
    size_t bytes_bits = num_chunk * sizeof(u2);
    size_t bytes_entries = num_chunk * sizeof(u4);
    size_t bytes_T_outlier = (sizeof(T) + sizeof(M)) * compact->h_num[0];
    size_t bytes_E_outlier = (sizeof(E) + sizeof(M)) * h_pbk_brnum[0];
    pbk_bytes = bytes_bitstream + bytes_tree_IDs + bytes_bits + bytes_entries + bytes_T_outlier +
                bytes_E_outlier;

    if (not print) return;
    printf(
        "output contains:\n"
        "| segments  |    uints | #byte |    bytes |\n"
        "| --------- | -------- | ----- | -------- |\n"
        "| bitstream | %8lu |     %u | %8lu |\n"
        "| tree IDs  | %8u |     %u | %8lu |\n"
        "| bits      | %8u |     %u | %8lu |\n"
        "| entries   | %8u |     %u | %8lu |\n"
        "| T outlier | %8u |     %u | %8lu |\n"
        "| E outlier | %8u |     %u | %8lu |\n",
        h_pbk_loc[0], (u4)sizeof(u4), bytes_bitstream,                    //
        num_chunk, (u4)sizeof(u1), bytes_tree_IDs,                        //
        num_chunk, (u4)sizeof(u2), bytes_bits,                            //
        num_chunk, (u4)sizeof(u4), bytes_entries,                         //
        compact->h_num[0], (u4)(sizeof(T) + sizeof(M)), bytes_T_outlier,  //
        h_pbk_brnum[0], (u4)(sizeof(E) + sizeof(M)), bytes_E_outlier      //
    );

    printf("bytes uncompressed  :  %lu\n", sizeof(T) * len);
    printf("bytes compressed    :  %lu\n", pbk_bytes);
    printf("compression ratio   :  %.2fx\n", sizeof(T) * len * 1.0 / pbk_bytes);
  }
};

}  // namespace psz
