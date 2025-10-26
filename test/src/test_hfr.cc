#include <cuda_runtime.h>

#include "c_type.h"
#include "hf_impl.hh"
#include "mem/cxx_memobj.h"
#include "rs_merge.hh"

template <typename T>
using memobj = _portable::memobj<T>;

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
  using Buf = typename phf::HuffmanCodec<T>::Buf;
  using HFR = HFReVISIT_config<T, Magnitude, ReduceTimes>;
  using ALT = testing_alt_code<ReduceTimes>;
  auto slab_size = HFR::ChunkSize;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t inlen = slab_size;
  size_t bklen = 1024;

  auto buf = std::make_unique<Buf>(inlen, bklen, 1, true);

  /* input  */ memobj<T> in(slab_size, "input");
  in.control({MallocHost, Malloc});

  for (auto i = 0; i < buf->bklen; i++) {
    // auto w = HuffmanWord<4>(0b1101, 4);
    // cout << bitset<32>(w.to_uint()) << endl;;
    auto w = HuffmanWord<4>(0b1, 1);
    buf->h_bk4[i] = w.to_uint();
  }
  // for (auto i = 0; i < bk.len(); i++) cout << bk.hat(i) << endl;
  memcpy_allkinds<H2D>(buf->d_bk4.get(), buf->h_bk4.get(), buf->bklen);

  phf::module::GPU_HFReVISIT_encode<T, Magnitude, ReduceTimes, false, u4>(
      {in.dptr(), in.len()}, {buf->h_bk4.get(), (u2)buf->bklen, ALT::alt_code, ALT::alt_bitcount},
      buf->dense_space(1), buf->sparse_space(), stream);

  cudaStreamSynchronize(stream);

  auto h_dn_bitcount = MAKE_UNIQUE_HOST(u4, buf->HFR_nchunk);
  memcpy_allkinds<D2H>(h_dn_bitcount.get(), buf->d_dn_bitcount.get(), buf->HFR_nchunk);

  // by default there is no host-side buf for dn_bitcount
  printf("dn_bitcount[0]: %d\n", h_dn_bitcount[0]);

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