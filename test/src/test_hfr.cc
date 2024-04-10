#include <cuda_runtime.h>

#include "detail/hfbuf.inl"
#include "hfword.hh"
#include "mem/definition.hh"
#include "mem/memobj.hh"
#include "rs_merge.hxx"

// using T = u4;
using namespace portable;

template <int ReduceTimes, typename Hf = u4>
struct testing_alt_code;

template <>
struct testing_alt_code<4, u4> {
  static const u4 alt_code = (0b1111111111111111 << 16);
  static const u4 alt_bitcount = 16;
};

template <>
struct testing_alt_code<3, u4> {
  static const u4 alt_code = (0b11111111u << 24);
  static const u4 alt_bitcount = 8;
};

template <>
struct testing_alt_code<2, u4> {
  static const u4 alt_code = (0b1111u << 28);
  static const u4 alt_bitcount = 4;
};

template <typename T, int Magnitude = 12, int ReduceTimes = 4>
void TEST_single_block_v2()
{
  using BUF = typename phf::HuffmanCodec<T>::internal_buffer;
  using HFR = HFReVISIT_config<T, Magnitude, ReduceTimes>;
  using ALT = testing_alt_code<ReduceTimes>;
  auto slab_size = HFR::ChunkSize;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t inlen = slab_size;
  size_t bklen = 1024;

  auto buf = std::make_unique<BUF>(inlen, bklen, 1, true);

  /* input  */ memobj<T> in(slab_size, "input");
  in.control({MallocHost, Malloc});

  for (auto i = 0; i < buf->bk4->len(); i++) {
    // auto w = HuffmanWord<4>(0b1101, 4);
    // cout << bitset<32>(w.to_uint()) << endl;;
    auto w = HuffmanWord<4>(0b1, 1);
    buf->bk4->hat(i) = w.to_uint();
  }
  // for (auto i = 0; i < bk.len(); i++) cout << bk.hat(i) << endl;
  buf->bk4->control({H2D});

  phf::cuhip::GPU_HFReVISIT_encode<T, Magnitude, ReduceTimes, false, u4>(
      {in.dptr(), in.len()},
      {buf->bk4->array1_d(), ALT::alt_code, ALT::alt_bitcount},
      buf->dense_space(1), buf->sparse_space(), stream);

  cudaStreamSynchronize(stream);

  buf->dn_bitstream->control({D2H});
  buf->dn_bitcount->control({D2H});

  // by default there is no host-side buf for dn_bitcount
  printf(
      "dn_bitcount[0]: %d\n",
      buf->dn_bitcount->control({MallocHost, D2H})->hat(0));

  // auto count = (dn_bitcount.hat(0) - 1) / 32 + 1;
  // for (auto i = 0; i < count; i++)
  //   cout << i << "\t" << std::bitset<32>(dn_out.hat(i)) << endl;

  cudaStreamDestroy(stream);
}

int main()
{
  TEST_single_block_v2<u2, 12, 4>();
  return 0;
}