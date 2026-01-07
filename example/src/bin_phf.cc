#include <cstdint>
#include <string>

#include "detail/compare.hh"
#include "hf.h"
#include "hf_hl.hh"
#include "kernel/hist.hh"
#include "kernel/predictor.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"
#include "utils/print_arr.hh"

namespace utils = _portable::utils;
using std::cout;
using std::endl;

using B = uint8_t;
using F = u4;

string fname;
bool dump_book{false}, use_revisit{false};
int sublen, pardeg;
uint8_t* d_compressed;
float time_hist;
size_t outlen;
cudaStream_t stream;
float time_encode = (float)INT_MAX;
float time_decode = (float)INT_MAX;
int which_test = 1;
phf_header header;

#define PEEK_DATA                        \
  printf("peeking data, 20 elements\n"); \
  psz::peek_data<E>(h_oridata, 20), printf("\n");

#define PEEK_XDATA                                                \
  printf("peeking xdata, 20 elements\n");                         \
  memcpy_allkinds<E, D2H>(h_decomp, d_decomp, len), printf("\n"); \
  psz::peek_data<E>(h_decomp, 20), printf("\n");

#define CHECK_INTEGRITY                                                               \
  auto identical =                                                                    \
      psz::module::GPU_identical(d_decomp.get(), d_oridata.get(), sizeof(E), len, 0); \
  printf("%s\n", identical ? ">>>>  IDENTICAL" : "!!!!  ERROR: DIFFERENT");

#define MALLOC_BUFFERS                                                 \
  auto d_oridata = GPU_make_unique(malloc_d<E>(len), GPU_DELETER_D()); \
  auto h_oridata = GPU_make_unique(malloc_h<E>(len), GPU_DELETER_H()); \
  auto d_decomp = GPU_make_unique(malloc_d<E>(len), GPU_DELETER_D());  \
  auto h_decomp = GPU_make_unique(malloc_h<E>(len), GPU_DELETER_H());  \
  auto d_hist = GPU_make_unique(malloc_d<F>(bklen), GPU_DELETER_D());  \
  auto h_hist = GPU_make_unique(malloc_h<F>(bklen), GPU_DELETER_H());

#define LOAD_FILE                                       \
  utils::fromfile(fname.c_str(), h_oridata.get(), len); \
  memcpy_allkinds<H2D>(d_oridata.get(), h_oridata.get(), len);

#define PREPARE   \
  MALLOC_BUFFERS; \
  LOAD_FILE;      \
  cudaStreamCreate(&stream);

#define CLEANUP cudaStreamDestroy(stream);

#define PRINT_REPORT                                    \
  print_GBps<E>(len, time_encode, "hf_encode");         \
  print_GBps<u1>(outlen, time_decode, "hf_decode");     \
  printf("Huffman in  bytes:\t%lu\n", len * sizeof(E)); \
  printf("Huffman out bytes:\t%lu\n", outlen);          \
  printf("Huffman CR (out/in):\t%.2lf\n", len * sizeof(E) * 1.0 / outlen);

#define PRINT_REPORT_RUN2                               \
  printf("Huffman in  bytes:\t%lu\n", len * sizeof(E)); \
  printf("Huffman out bytes:\t%lu\n", outlen);          \
  printf("Huffman CR (out/in):\t%.2lf\n", len * sizeof(E) * 1.0 / outlen);

namespace {
void print_tobediscarded_info(float time_in_ms, string fn_name)
{
  auto title = "[psz::info::discard::" + fn_name + "]";
  printf("%s time (ms): %.6f\n", title.c_str(), time_in_ms);
}

template <typename T>
float print_GBps(size_t len, float time_in_ms, string fn_name)
{
  auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto GiBps = len * sizeof(T) * 1.0 / B_to_GiB / (time_in_ms / 1000);
  auto title = "[psz::info::res::" + fn_name + "]";
  printf("%s %.2f GiB/s at %.6f ms\n", title.c_str(), GiBps, time_in_ms);
  return GiBps;
}

struct Arguments {
  std::string fname;
  int x = 0, y = 0, z = 0;
  int bklen = 0;
  std::string type = "u1";  // default
  bool use_revisit = false;
  bool dump_book = false;
  int which_test = 1;

  bool parse(int argc, char** argv)
  {
    if (argc < 6) {
      print_usage(argv[0]);
      return false;
    }

    fname = argv[1];
    x = std::atoi(argv[2]);
    y = std::atoi(argv[3]);
    z = std::atoi(argv[4]);
    bklen = std::atoi(argv[5]);

    for (int i = 6; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--fast") { use_revisit = true; }
      else if (arg == "--dump-book") {
        dump_book = true;
      }
      else if (arg == "--type" and i + 1 < argc) {
        type = argv[++i];
      }
      else if (arg == "--test" and i + 1 < argc) {
        which_test = std::atoi(argv[++i]);
      }
      else {
        printf("Unknown or incomplete argument: %s\n", arg.c_str());
        print_usage(argv[0]);
        return false;
      }
    }

    return true;
  }

  size_t total_len() const { return static_cast<size_t>(x) * y * z; }

  void print_usage(const char* progname) const
  {
    printf(
        "usage:\n"
        "  %s  /path/to/data  X  Y  Z  bklen  "
        "[--fast true|false]  [--type u1|u2|u4]  [--dump-book true|false]  [--test 1|2]\n",
        progname);
  }
};

}  // namespace

template <typename E, typename H = u4>
void hf_run_3(std::string fname, size_t const len, size_t const bklen = 1024)
{
  PREPARE;

  auto buf = new phf::Buf<E>(len, bklen, -1, true);
  int hist_generic_grid_dim, hist_generic_block_dim, shmem_use, repeat;
  psz::module::GPU_histogram_generic<E>::init(
      len, bklen, hist_generic_grid_dim, hist_generic_block_dim, shmem_use, repeat);
  psz::module::GPU_histogram_generic<E>::kernel(
      d_oridata.get(), len, d_hist.get(), bklen, hist_generic_grid_dim, hist_generic_block_dim,
      shmem_use, repeat, stream);
  memcpy_allkinds_async<D2H>(h_hist.get(), d_hist.get(), bklen);
  sync_by_stream(stream);

  phf::high_level<E>::build_book(buf, h_hist.get(), bklen, stream);

  for (auto i = 0; i < 10; i++) {
    phf::high_level<E>::encode_ReVISIT_lite(
        buf, d_oridata.get(), len, &d_compressed, &outlen, header, stream);
    phf::high_level<E>::decode(buf, header, d_compressed, d_decomp.get(), stream);
  }

  CHECK_INTEGRITY;
  PRINT_REPORT_RUN2;
  CLEANUP;
}

template <typename E, typename H = u4>
void hf_run_2(std::string fname, size_t const len, size_t const bklen = 1024)
{
  PREPARE;

  auto buf = new phf::Buf<E>(len, bklen);
  int hist_generic_grid_dim, hist_generic_block_dim, shmem_use, repeat;
  psz::module::GPU_histogram_generic<E>::init(
      len, bklen, hist_generic_grid_dim, hist_generic_block_dim, shmem_use, repeat);
  psz::module::GPU_histogram_generic<E>::kernel(
      d_oridata.get(), len, d_hist.get(), bklen, hist_generic_grid_dim, hist_generic_block_dim,
      shmem_use, repeat, stream);
  memcpy_allkinds_async<D2H>(h_hist.get(), d_hist.get(), bklen);
  sync_by_stream(stream);

  phf::high_level<E>::build_book(buf, h_hist.get(), bklen, stream);

  for (auto i = 0; i < 10; i++) {
    phf::high_level<E>::encode(buf, d_oridata.get(), len, &d_compressed, &outlen, header, stream);
    phf::high_level<E>::decode(buf, header, d_compressed, d_decomp.get(), stream);
  }

  CHECK_INTEGRITY;
  PRINT_REPORT_RUN2;
  CLEANUP;
}

int main(int argc, char** argv)
{
  Arguments args;
  if (not args.parse(argc, argv)) { return 1; }

  size_t len = args.total_len();

  if (args.which_test == 1) {
    cout << "HF run version-1 is removed, exiting..." << args.type << endl;
  }
  else if (args.which_test == 2) {
    cout << "HF run version-2, input type: " << args.type << endl;

    if (args.type == "u1")
      hf_run_2<uint8_t>(args.fname, len, 256);
    else if (args.type == "u2")
      hf_run_2<uint16_t>(args.fname, len, args.bklen);
    else if (args.type == "u4")
      hf_run_2<uint32_t>(args.fname, len, args.bklen);
    else
      fprintf(stderr, "Unknown type: %s\n", args.type.c_str());
  }
  else {
    cout << "HF run version-3 (ReVISIT), input type: " << args.type << endl;

    if (args.type == "u1")
      hf_run_3<uint8_t>(args.fname, len, 256);
    else if (args.type == "u2")
      hf_run_3<uint16_t>(args.fname, len, args.bklen);
    else if (args.type == "u4")
      hf_run_3<uint32_t>(args.fname, len, args.bklen);
    else
      fprintf(stderr, "Unknown type: %s\n", args.type.c_str());
  }

  return 0;
}