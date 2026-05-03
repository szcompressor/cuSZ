#ifndef LC_GEN_LC_BUF_H
#define LC_GEN_LC_BUF_H

#include <algorithm>
#include <cstddef>

#include "cusz/type.h"
#include "mem/cxx_backends.h"

namespace psz {

struct LC_Buf {
 public:
  static constexpr size_t CHUNK_BYTES = 1024 * 16;

 private:
  size_t encoded_bytes_;
  size_t decoded_bytes_;
  size_t max_chunks_;

  GPU_unique_dptr<byte_t[]> d_encoded_;
  GPU_unique_dptr<byte_t[]> d_decoded_;
  GPU_unique_dptr<int[]> d_size_;
  GPU_unique_dptr<int[]> d_fullcarry_;
  GPU_unique_hptr<int[]> h_size_;

  static size_t chunk_count(size_t bytes) { return (bytes + CHUNK_BYTES - 1) / CHUNK_BYTES; }

  static size_t encoded_capacity(size_t input_bytes, bool need_align8)
  {
    const auto chunks = chunk_count(input_bytes);
    const auto base = 3 * sizeof(int) + chunks * sizeof(unsigned short) + chunks * CHUNK_BYTES;
    return need_align8 ? base + 7 : base;
  }

 public:
  LC_Buf(
      size_t tcms_input_bytes, size_t bitr_input_bytes, size_t rtr_input_bytes,
      size_t decoded_bytes_max)
  {
    encoded_bytes_ = std::max(
        {encoded_capacity(tcms_input_bytes, true), encoded_capacity(bitr_input_bytes, false),
         encoded_capacity(rtr_input_bytes, false)});
    decoded_bytes_ = decoded_bytes_max;
    max_chunks_ = std::max(
        {chunk_count(tcms_input_bytes), chunk_count(bitr_input_bytes), chunk_count(rtr_input_bytes),
         chunk_count(decoded_bytes_max)});

    d_encoded_ = MAKE_UNIQUE_DEVICE(byte_t, encoded_bytes_);
    d_decoded_ = MAKE_UNIQUE_DEVICE(byte_t, decoded_bytes_);
    d_size_ = MAKE_UNIQUE_DEVICE(int, 1);
    d_fullcarry_ = MAKE_UNIQUE_DEVICE(int, max_chunks_);
    h_size_ = MAKE_UNIQUE_HOST(int, 1);
  }

  ~LC_Buf() = default;

  byte_t* encoded_d() const { return d_encoded_.get(); }
  byte_t* decoded_d() const { return d_decoded_.get(); }
  int* size_d() const { return d_size_.get(); }
  int* fullcarry_d() const { return d_fullcarry_.get(); }
  int* size_h() const { return h_size_.get(); }

  size_t encoded_capacity() const { return encoded_bytes_; }
  size_t decoded_capacity() const { return decoded_bytes_; }
  size_t max_chunks() const { return max_chunks_; }
};

}  // namespace psz

#endif
