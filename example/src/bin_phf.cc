#include <cstdint>
#include <string>

#include "detail/compare.hh"
#include "hf.h"
#include "hf_hl.hh"
#include "kernel/hist.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"
#include "utils/print_arr.hh"

namespace utils = _portable::utils;
using std::string;

using F = u4;

namespace {

template <typename T>
void print_GBps(size_t len, float time_ms, const char* label)
{
  double GBps = (double)len * sizeof(T) / 1e9 / (time_ms * 1e-3);
  printf("[psz::info::res::%s] %.2f GB/s at %.4f ms\n", label, GBps, time_ms);
}

struct Arguments {
  string fname;
  int x = 0, y = 0, z = 0;
  int bklen = 1024;
  string type = "u2";
  bool use_hfr = false;

  bool parse(int argc, char** argv)
  {
    if (argc < 6) {
      print_usage(argv[0]);
      return false;
    }
    fname = argv[1];
    x     = std::atoi(argv[2]);
    y     = std::atoi(argv[3]);
    z     = std::atoi(argv[4]);
    bklen = std::atoi(argv[5]);

    for (int i = 6; i < argc; ++i) {
      string arg = argv[i];
      if (arg == "--hfr")
        use_hfr = true;
      else if (arg == "--hf")
        use_hfr = false;
      else if (arg == "--type" && i + 1 < argc)
        type = argv[++i];
      else {
        printf("unknown argument: %s\n", arg.c_str());
        print_usage(argv[0]);
        return false;
      }
    }
    return true;
  }

  size_t total_len() const { return (size_t)x * y * z; }

  void print_usage(const char* prog) const
  {
    printf(
        "usage: %s  /path/to/data  X  Y  Z  bklen"
        "  [--hfr|--hf]  [--type u1|u2|u4]\n",
        prog);
  }
};

}  // namespace

template <typename E>
void hf_run(const string& fname, size_t len, int bklen, bool use_hfr)
{
  printf(
      "[hf_run] codec=%s  len=%zu  bklen=%d\n",
      use_hfr ? "HFR" : "HF", len, bklen);

  auto h_data   = malloc_host<E>(len);
  auto d_data   = malloc_device<E>(len);
  auto d_decomp = malloc_device<E>(len);
  auto d_hist   = malloc_device<F>(bklen);
  auto h_hist   = malloc_host<F>(bklen);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  utils::fromfile(fname.c_str(), h_data, len);
  memcpy_allkinds_async<H2D>(d_data, h_data, len, stream);
  cudaStreamSynchronize(stream);

  int grid_dim, block_dim, shmem_use, repeat;
  psz::module::GPU_histogram_generic<E>::init(len, bklen, grid_dim, block_dim, shmem_use, repeat);
  psz::module::GPU_histogram_generic<E>::kernel(
      d_data, len, d_hist, bklen, grid_dim, block_dim, shmem_use, repeat, stream);
  memcpy_allkinds_async<D2H>(h_hist, d_hist, bklen, stream);
  cudaStreamSynchronize(stream);

  auto buf = new phf::Buf<E>(len, bklen, -1, use_hfr);
  phf::high_level<E>::build_book(buf, h_hist, bklen, stream);

  uint8_t* d_encoded = nullptr;
  size_t   encoded_len = 0;
  phf_header header{};

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  cudaEventRecord(t0, stream);
  if (use_hfr)
    phf::high_level<E>::encode_HFR(buf, d_data, len, &d_encoded, &encoded_len, header, stream);
  else
    phf::high_level<E>::encode(buf, d_data, len, &d_encoded, &encoded_len, header, stream);
  cudaEventRecord(t1, stream);
  cudaStreamSynchronize(stream);

  float ms_enc = 0;
  cudaEventElapsedTime(&ms_enc, t0, t1);

  cudaEventRecord(t0, stream);
  phf::high_level<E>::decode(buf, header, d_encoded, d_decomp, stream);
  cudaEventRecord(t1, stream);
  cudaStreamSynchronize(stream);

  float ms_dec = 0;
  cudaEventElapsedTime(&ms_dec, t0, t1);

  auto identical =
      psz::module::GPU_identical((void*)d_decomp, (void*)d_data, sizeof(E), len, stream);
  printf("%s\n", identical ? ">>>>  IDENTICAL" : "!!!!  ERROR: DIFFERENT");

  print_GBps<E>(len, ms_enc, "hf_encode");
  print_GBps<u1>(encoded_len, ms_dec, "hf_decode");
  printf("Huffman in  bytes: %zu\n", len * sizeof(E));
  printf("Huffman out bytes: %zu\n", encoded_len);
  printf("Huffman CR (in/out): %.2f\n", (double)(len * sizeof(E)) / encoded_len);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaStreamDestroy(stream);
  delete buf;
  free_host(h_data);
  free_device(d_data);
  free_device(d_decomp);
  free_device(d_hist);
  free_host(h_hist);
}

int main(int argc, char** argv)
{
  Arguments args;
  if (!args.parse(argc, argv)) return 1;

  size_t len = args.total_len();

  if (args.type == "u1")
    hf_run<uint8_t>(args.fname, len, 256, args.use_hfr);
  else if (args.type == "u2")
    hf_run<uint16_t>(args.fname, len, args.bklen, args.use_hfr);
  else if (args.type == "u4")
    hf_run<uint32_t>(args.fname, len, args.bklen, args.use_hfr);
  else
    fprintf(stderr, "unknown type: %s\n", args.type.c_str());

  return 0;
}
