#ifndef PSZ_MEM_BUF_LC_HH
#define PSZ_MEM_BUF_LC_HH

#include <cstddef>

#include "cusz/type.h"
#include "mem/cxx_backends.h"
#include "mem/plan.h"

namespace psz {

struct LC_Buf : _ptb::buf_base {
 public:
  static constexpr size_t CHUNK_BYTES = 1024 * 16;

 private:
  enum : int { ENCODED, DECODED, SIZE, FULLCARRY };

  size_t encoded_bytes_;
  size_t decoded_bytes_;
  size_t max_chunks_;

  GPU_unique_dptr<u1[]> d_data, d_state;
  byte_t* wired_encoded_ = nullptr;
  GPU_unique_hptr<int[]> h_size_;

  static size_t chunk_count(size_t bytes) { return (bytes + CHUNK_BYTES - 1) / CHUNK_BYTES; }

  struct dims {
    size_t encoded_bytes, decoded_bytes, max_chunks;
  };

  static dims plan_lc(psz_codec c, size_t encoded_in, size_t chunked_in_max, size_t decoded_max)
  {
    return {
        encoded_in ? encoded_capacity(encoded_in, needs_align8(c)) : 0, decoded_max,
        chunk_count(chunked_in_max)};
  }

  static tables frames(dims const& x)
  {
    return {
        // arrays
        {{ENCODED, U1, x.encoded_bytes}, {DECODED, U1, x.decoded_bytes}},
        // metadata
        {{SIZE, I4, 1}, {FULLCARRY, I4, x.max_chunks}}};
  }

  LC_Buf(dims const& x) :
      buf_base(frames(x)),
      encoded_bytes_(x.encoded_bytes),
      decoded_bytes_(x.decoded_bytes),
      max_chunks_(x.max_chunks)
  { h_size_ = MAKE_UNIQUE_HOST(int, 1); }

 public:
  static size_t encoded_capacity(size_t input_bytes, bool need_align8)
  {
    const auto chunks = chunk_count(input_bytes);
    const auto base = 3 * sizeof(int) + chunks * sizeof(unsigned short) + chunks * CHUNK_BYTES;
    return need_align8 ? base + 7 : base;
  }

  static bool needs_align8(psz_codec c)
  { return c == psz_codec::LC_TCMS or c == psz_codec::LC_DRH; }

  LC_Buf(psz_codec c, size_t encoded_in, size_t chunked_in_max, size_t decoded_max) :
      LC_Buf(plan_lc(c, encoded_in, chunked_in_max, decoded_max))
  {
  }

  ~LC_Buf() override = default;

  void init() override
  {
    d_data = MAKE_UNIQUE_DEVICE(u1, data().bytes());
    d_state = MAKE_UNIQUE_DEVICE(u1, state().bytes());
    attach(d_data.get(), d_state.get());
  }

  void reset(void*) override {}  // simply not needed

  void wire_encoded(byte_t* external) { wired_encoded_ = external; }
  byte_t* encoded_d() const
  { return wired_encoded_ ? wired_encoded_ : data().ptr_of<byte_t>(ENCODED); }
  byte_t* decoded_d() const { return data().ptr_of<byte_t>(DECODED); }
  int* size_d() const { return state().ptr_of<int>(SIZE); }
  int* fullcarry_d() const { return state().ptr_of<int>(FULLCARRY); }
  int* size_h() const { return h_size_.get(); }

  size_t encoded_capacity() const { return encoded_bytes_; }
  size_t decoded_capacity() const { return decoded_bytes_; }
  size_t max_chunks() const { return max_chunks_; }
};

}  // namespace psz

#endif
