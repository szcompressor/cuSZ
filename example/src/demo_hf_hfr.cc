#include <cuda_runtime.h>

#include "detail/hfbuf.inl"
#include "hfword.hh"
#include "kernel/hist.hh"
#include "mem/definition.hh"
#include "mem/memobj.hh"
#include "mem/multibackend.hh"
#include "rs_merge.hh"
#include "utils/io.hh"

float _time;

template <typename E = uint16_t, int Magnitude = 10, int ReduceTimes = 4>
void TEST_using_codec(string fname_base, size_t const len, uint16_t const bklen)
{
  using FREQ = u4;

  auto h_data = malloc_host<E>(len);
  auto d_data = malloc_device<E>(len);
  auto h_hist = malloc_host<FREQ>(len);

  cudaStream_t stream_ref;
  cudaStreamCreate(&stream_ref);
  auto h_decoded_ref = malloc_host<E>(len);
  auto d_decoded_ref = malloc_device<E>(len);
  u1* encoded_ref;
  size_t encoded_len_ref;

  cudaStream_t stream_hfr;
  cudaStreamCreate(&stream_hfr);
  auto h_decoded_hfr = malloc_host<E>(len);
  auto d_decoded_hfr = malloc_device<E>(len);
  u1* encoded_hfr;
  size_t encoded_len_hfr;

  psz::module::SEQ_histogram_Cauchy_v2(h_data, len, h_hist, bklen, &_time);

  _portable::utils::fromfile(fname_base + ".quant", &h_data, len);
  memcpy_allkinds<E, H2D>(d_data, h_data, len);

  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  constexpr auto Sublen = 1 << Magnitude;
  auto pardeg = div(len, Sublen);

  phf::HuffmanCodec<E> codec_ref(len, bklen, pardeg);
  codec_ref.buildbook(h_hist, stream_ref);
  codec_ref.encode(false, d_data, len, &encoded_ref, &encoded_len_ref, stream_ref);
  codec_ref.decode(encoded_ref, d_decoded_ref, stream_ref);

  phf::HuffmanCodec<E> codec_hfr(len, bklen, pardeg);
  codec_hfr.buildbook(h_hist, stream_hfr);
  codec_hfr.encode(true, d_data, len, &encoded_hfr, &encoded_len_hfr, stream_hfr);
  codec_hfr.decode(encoded_ref, d_decoded_hfr, stream_hfr);

  free_host(h_hist);
  free_host(h_data);
  free_device(h_data);

  cudaStreamDestroy(stream_ref);
  cudaStreamDestroy(stream_hfr);
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    // clang-format off
    printf(
        "PROG  /path/to/data  X  Y  Z  bklen  [type: u1,u2,u4] \n"
        "0     1              2  3  4  5      [6:optional]     \n");
    // clang-format on
    exit(0);
  }
  else {
    auto fname_base = std::string(argv[1]);
    auto x = atoi(argv[2]), y = atoi(argv[3]), z = atoi(argv[4]);
    auto len = x * y * z;
    auto bklen = atoi(argv[5]);

    auto type = string("u1");
    if (argc == 7) type = std::string(argv[6]);

    if (type == "u1") {
      printf("REVERT bklen to 256 for u1-type input.\n");
      TEST_using_codec<u1, 10>(fname_base, len, 256);
    }
    else {
      if (type == "u2")
        TEST_using_codec<u2, 10>(fname_base, len, bklen);
      else if (type == "u4")
        TEST_using_codec<u4, 10>(fname_base, len, bklen);
      else
        TEST_using_codec<u4, 10>(fname_base, len, bklen);
    }

    return 0;
  }
}