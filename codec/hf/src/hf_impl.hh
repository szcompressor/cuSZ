#ifndef HF_WORD_HH
#define HF_WORD_HH

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "c_type.h"
#include "hf.h"
#include "mem/cxx_array.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <int WIDTH>
struct HuffmanWord;

using PW4 = HuffmanWord<4>;
using PW8 = HuffmanWord<8>;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// clang-format off
template <int WIDTH> constexpr int FIELD_CODE();
template <> constexpr int FIELD_CODE<4>() { return 27; }
template <> constexpr int FIELD_CODE<8>() { return 58; }
template <int WIDTH> constexpr int BITWIDTH() { return WIDTH * 8; }
template <int WIDTH> constexpr int FIELD_BITCOUNT() { return BITWIDTH<WIDTH>() - FIELD_CODE<WIDTH>(); }
template <int WIDTH> constexpr int OUTLIER_CUTOFF() { return FIELD_CODE<WIDTH>() + 1; }

namespace {
  template <int W> struct HFtype;
  template <> struct HFtype<4> { typedef u4 type; };
  template <> struct HFtype<8> { typedef u8 type; };
}
// clang-format on

// [psz::caveat] Direct access leads to misaligned GPU addr.
// MSB | log2(32)=5 | max: 27; var-len prefix-code, right aligned |
// MSB | log2(64)=6 | max: 58; var-len prefix-code, right aligned |
template <int WIDTH>
struct HuffmanWord {
  static constexpr int W = WIDTH;
  static constexpr int BITWIDTH = ::BITWIDTH<W>();
  static constexpr int FIELD_CODE = ::FIELD_CODE<W>();
  static constexpr int FIELD_BITCOUNT = ::FIELD_BITCOUNT<W>();
  static constexpr int OUTLIER_CUTOFF = ::OUTLIER_CUTOFF<W>();
  using Hf = typename HFtype<W>::type;

  Hf prefix_code : FIELD_CODE;   // low 27 (58 for u8) bits
  Hf bitcount : FIELD_BITCOUNT;  // high 5 (6 for u8) bits

  HuffmanWord(Hf _prefix_code, Hf _bitcount)
  {
    prefix_code = _prefix_code;
    bitcount = _bitcount;
  }

  Hf to_uint() { return *reinterpret_cast<Hf*>(this); }
};

// MSB | max: 27; var-len prefix-code, left aligned | optional log2(32)=5 |
// MSB | max: 58; var-len prefix-code, left aligned | optional log2(64)=6 |
template <int W>
struct HuffmanWordLeftAlign {
  uint32_t bitcount : HuffmanWord<4>::FIELD_BITCOUNT;  // low 5 bits
  uint32_t prefix_code : HuffmanWord<4>::FIELD_CODE;   // high 27 bits
};

template <int W>
void rightalign_to_leftalign(HuffmanWord<W>& in, HuffmanWordLeftAlign<W>& out)
{
  out.bitcount = in.bitcount;
  out.prefix_code = in.prefix_code << (HuffmanWord<W>::FIELD_CODE - in.bitcount);
}

// for impl1

struct alignas(8) node_t {
  struct node_t *left, *right;
  size_t freq;
  char t;  // in_node:0; otherwise:1
  union {
    uint32_t c;
    uint32_t symbol;
  };
};

typedef struct node_t* node_list;

typedef struct alignas(8) hfserial_tree {
  uint32_t state_num;
  uint32_t all_nodes;
  struct node_t* pool;
  node_list *qqq, *qq;  // the root node of the hfserial_tree is qq[1]
  int n_nodes;          // n_nodes is for compression
  int qend;
  uint64_t** code;
  uint8_t* cout;
  int n_inode;  // n_inode is for decompression
} hfserial_tree;
typedef hfserial_tree HuffmanTree;

template <typename H>
void phf_CPU_build_codebook_v1(u4* freq, uint16_t bklen, H* book);

// for impl2

struct phf_node {
  u4 symbol;
  u4 freq;
  phf_node *left, *right;

  phf_node(u4 symbol, u4 freq, phf_node* left = nullptr, phf_node* right = nullptr) :
      symbol(symbol), freq(freq), left(left), right(right)
  {
  }
};

struct phf_cmp_node {
  bool operator()(phf_node* left, phf_node* right) { return left->freq > right->freq; }
};

template <class NodeType, int WIDTH>
class alignas(8) phf_stack {
  static const int MAX_DEPTH = HuffmanWord<WIDTH>::FIELD_CODE;
  NodeType* _a[MAX_DEPTH];
  u8 saved_path[MAX_DEPTH];
  u8 saved_length[MAX_DEPTH];
  u8 depth = 0;

 public:
  static NodeType* top(phf_stack* s);

  template <typename T>
  static void push(phf_stack* s, NodeType* n, T path, T len);

  template <typename T>
  static NodeType* pop(phf_stack* s, T* path_to_restore, T* length_to_restore);

  template <typename H>
  static void inorder_traverse(NodeType* root, H* book);
};

template <typename H>
void phf_CPU_build_codebook_v2(u4* freq, size_t const bklen, H* book);

template <typename E, typename H>
class hf_space {
 public:
  static const int TYPE_BITS = sizeof(H) * 8;

  static u4 space_bytes(int const bklen)
  {
    return sizeof(H) * (3 * bklen) + sizeof(u4) * (4 * TYPE_BITS) + sizeof(E) * bklen;
  }

  static u4 revbook_bytes(int const bklen)
  {
    return sizeof(u4) * (2 * TYPE_BITS) + sizeof(E) * bklen;
  }

  static u4 revbook_offset(int const bklen)
  {
    return sizeof(H) * (3 * bklen) + sizeof(u4) * (2 * TYPE_BITS);
  }
};

template <typename E, typename H>
int canonize_on_gpu(uint8_t* bin, uint32_t bklen, void* stream);

template <typename E, typename H>
int canonize(uint8_t* bin, uint32_t const bklen);

///////////////////////////////////////////////////////////////////////////////

template <typename E = uint32_t, typename H = uint32_t>
class hf_canon_reference {
 private:
  H *_icb, *_ocb, *_canon;
  int *_numl, *_iterby, *_first, *_entry;
  E* _keys;

 public:
  // public var
  uint16_t const booklen;
  static const auto TYPE_BITS = sizeof(H) * 8;

  // public fn
  hf_canon_reference(uint16_t booklen) : booklen(booklen) { init(); }
  ~hf_canon_reference()
  {
    // delete[] _icb,
    delete[] _ocb, delete[] _canon;
    delete[] _keys;
    delete[] _numl, delete[] _iterby, delete[] _first, delete[] _entry;
  }
  void init()
  {
    // _icb = new H[booklen], memset(_icb, 0, sizeof(H) * booklen);
    _ocb = new H[booklen], memset(_ocb, 0, sizeof(H) * booklen);
    _canon = new H[booklen], memset(_canon, 0, sizeof(H) * booklen);
    _numl = new int[TYPE_BITS], memset(_numl, 0, sizeof(int) * TYPE_BITS);
    _iterby = new int[TYPE_BITS], memset(_iterby, 0, sizeof(int) * TYPE_BITS);
    // revbook involves the below arrays
    _first = new int[TYPE_BITS], memset(_first, 0, sizeof(int) * TYPE_BITS);
    _entry = new int[TYPE_BITS], memset(_entry, 0, sizeof(int) * TYPE_BITS);
    _keys = new E[booklen], memset(_keys, 0, sizeof(E) * booklen);
  }

  // accessor
  H*& input_bk() { return _icb; }
  H* output_bk() { return _ocb; }
  H* canon() { return _canon; }
  E* keys() { return _keys; }
  int* numl() { return _numl; }
  int* iterby() { return _iterby; }
  int* first() { return _first; }
  int* entry() { return _entry; }

  H& input_bk(int i) { return _icb[i]; }
  H& output_bk(int i) { return _ocb[i]; }
  H& canon(int i) { return _canon[i]; }
  E& keys(int i) { return _keys[i]; }
  int& numl(int i) { return _numl[i]; }
  int& iterby(int i) { return _iterby[i]; }
  int& first(int i) { return _first[i]; }
  int& entry(int i) { return _entry[i]; }
  // run
  int canonize();
};

namespace phf {

// template <typename T>
// using array = _portable::array1<T>;

// template <typename T>
// using sparse = _portable::compact_array1<T>;

template <typename Hf>
struct [[deprecated]] book {
  Hf* bk;
  u2 bklen;
  Hf const alt_prefix_code;  // even if u8 can use short u4 internal
  u4 const alt_bitcount;
};

template <typename Hf>
struct [[deprecated]] dense {
  Hf* const out;
  u4* bits;
  size_t n_part;
};

struct par_config {
  const size_t sublen;
  const size_t pardeg;
};

}  // namespace phf

template <typename T, typename H>
void phf_GPU_build_canonized_codebook(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook, int const revbook_nbyte,
    float* time, void* = nullptr);

template <typename E, typename H = uint32_t>
[[deprecated("use phf_CPU_build_canonized_codebook_v2")]] void phf_CPU_build_canonized_codebook_v1(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook, int const revbook_bytes,
    float* time);

template <typename E, typename H = uint32_t>
void phf_CPU_build_canonized_codebook_v2(
    uint32_t* freq, int const bklen, uint32_t* bk4, uint8_t* revbook, int const revbook_bytes,
    float* time = nullptr);

namespace phf::cuhip {

// @brief a namespace-like class for batch template instantiations; a rewrite of
// hf_kernels.{hh,cc}; all the included wrapped kernels/methods are `static`
// @tparam E input type
// @tparam H intermediate type for Huffman coding
template <typename E, typename H>
class modules {
  // metadata, e.g., saved index for parallel operations
  using M = PHF_METADATA;

 public:
  static void GPU_coarse_encode_phase1(
      E* in_data, const size_t data_len, H* in_book, const u4 book_len, const int numSMs,
      H* out_bitstream, void* stream);

  static void GPU_coarse_encode_phase2(
      H* in_data, const size_t data_len, phf::par_config hfpar, H* deflated, M* par_nbit,
      M* par_ncell, void* stream);

  static void GPU_fine_encode_phase1_2(
      E* in, const size_t len, H* book, const u4 bklen, H* bitstream, M* par_nbit, M* par_ncell,
      const u4 nblock, E* brval, u4* bridx, u4* brnum, void* stream);

  static void GPU_coarse_encode_phase3_sync(
      phf::par_config hfpar, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell,
      M* d_par_entry, M* h_par_entry, size_t* outlen_nbit, size_t* outlen_ncell,
      float* time_cpu_time, void* stream);

  static void GPU_coarse_encode_phase4(
      H* in_buf, const size_t len, M* par_entry, M* par_ncell, phf::par_config hfpar, H* bitstream,
      const size_t max_bitstream_len, void* stream);

  static void GPU_coarse_encode(
      E* in_data, size_t data_len, H* in_book, u4 book_len, int numSMs, phf::par_config hfpar,
      // internal buffers
      H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
      M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len,
      // output
      size_t* out_total_nbit, size_t* out_total_ncell, void* stream);

  static void GPU_fine_encode(
      E* in_data, size_t data_len, H* in_book, u4 book_len, phf::par_config hfpar,
      // internal buffers
      H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
      M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len, E* d_brval, u4* d_bridx,
      u4* d_brnum,
      // output
      size_t* out_total_nbit, size_t* out_total_ncell, void* stream);

  static void GPU_coarse_decode(
      H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len, M* in_par_nbit,
      M* in_par_entry, size_t const sublen, size_t const pardeg, E* out_decoded, void* stream);

  static void GPU_scatter(E* val, u4* idx, const u4 h_num, E* out, void* stream);
};

}  // namespace phf::cuhip

#endif /* HF_WORD_HH */
