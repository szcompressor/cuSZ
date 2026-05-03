#include <cupti_activity.h>

// CUpti_ActivityKernel* v5 onward uses the same start-end pattern.
#if CUPTI_API_VERSION >= 18  // CUDA 12.x
using CUpti_AK = CUpti_ActivityKernel11;
#elif CUPTI_API_VERSION >= 17  // CUDA 11.6–11.8
using CUpti_AK = CUpti_ActivityKernel9;
#elif CUPTI_API_VERSION >= 15  // CUDA 11.0–11.5
using CUpti_AK = CUpti_ActivityKernel8;
#elif CUPTI_API_VERSION >= 13  // CUDA 10.x
using CUpti_AK = CUpti_ActivityKernel6;
#else
using CUpti_AK = CUpti_ActivityKernel5;
#endif

#include <atomic>
#include <cstdint>
#include <string>

#include "compare.hh"
#include "hf.h"
#include "hf_buf.hh"  // needed for Buf instantiation
#include "hf_hl.hh"
#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"

namespace utils = _portable::utils;
using std::string;

using F = u4;

static bool g_cupti_active = false;

static std::atomic<uint64_t> s_kernel_ns{0};

static void CUPTIAPI phf_buf_req(uint8_t** buf, size_t* sz, size_t* maxRec)
{
  *sz = 1u << 20;  // 1 MB
  *buf = (uint8_t*)malloc(*sz);
  *maxRec = 0;
}

static void CUPTIAPI phf_buf_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t validSz)
{
  CUpti_Activity* rec = nullptr;
  while (cuptiActivityGetNextRecord(buf, validSz, &rec) == CUPTI_SUCCESS) {
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = (CUpti_AK*)rec;
      if (k->start != 0 and k->end >= k->start) s_kernel_ns += k->end - k->start;
    }
  }
  free(buf);
}

// cudaEvent-based timer: stream wall-clock.
struct CudaEventTimer {
  cudaEvent_t e0, e1;
  CudaEventTimer() { cudaEventCreate(&e0), cudaEventCreate(&e1); }
  ~CudaEventTimer() { cudaEventDestroy(e0), cudaEventDestroy(e1); }
  void start(cudaStream_t s) { cudaEventRecord(e0, s); }
  double stop_ms(cudaStream_t s)
  {
    cudaEventRecord(e1, s), cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    return ms;
  }
};

struct KernelTimer {
  // CUPTI path: sums actual kernel time (like nsys kernel timeline)
  // cudaEvent path: stream wall-clock
  void start(cudaStream_t s)
  {
    if (g_cupti_active) {
      s_kernel_ns = 0;
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    }
    else {
      ev_.start(s);
    }
  }
  double stop_ms(cudaStream_t s)
  {
    if (g_cupti_active) {
      cudaDeviceSynchronize();
      cuptiActivityFlushAll(0);
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
      return s_kernel_ns.load() * 1e-6;
    }
    else {
      return ev_.stop_ms(s);
    }
  }
  CudaEventTimer ev_;
};

namespace {

template <typename T>
float print_GiBps(size_t len, float time_ms, string label)
{
  auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto bytes = len * sizeof(T);
  auto GiBps = bytes * 1.0 / B_to_GiB / (time_ms / 1000);
  printf("%s:\t%.2f GiB/s, %zu bytes at %.4f ms\n", label.c_str(), GiBps, bytes, time_ms);
  return GiBps;
}

struct Arguments {
  string fname;
  int x = 0, y = 0, z = 0;
  int bklen = 1024;
  string type = "u2";
  bool use_hfr = false;
  int repeat = 1;         // number of encode/decode iterations; first run is warmup when repeat>1
  bool use_cupti = true;  // --timer cupti (default) | --timer event

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
      string arg = argv[i];
      if (arg == "--hfr")
        use_hfr = true;
      else if (arg == "--hf")
        use_hfr = false;
      else if (arg == "--type" and i + 1 < argc)
        type = argv[++i];
      else if (arg == "--repeat" and i + 1 < argc)
        repeat = std::atoi(argv[++i]);
      else if (arg == "--timer" and i + 1 < argc) {
        string t = argv[++i];
        if (t == "cupti")
          use_cupti = true;
        else if (t == "event")
          use_cupti = false;
        else {
          printf("unknown --timer value: %s  (choices: cupti, event)\n", t.c_str());
          print_usage(argv[0]);
          return false;
        }
      }
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
        "  [--hfr|--hf]  [--type u1|u2|u4]  [--repeat N]"
        "  [--timer cupti|event]\n",
        prog);
  }
};

}  // namespace

template <typename E>
void hf_run(const string& fname, size_t len, int bklen, bool use_hfr, int repeat)
{
  auto h_data = MAKE_UNIQUE_HOST(E, len);
  auto d_data = MAKE_UNIQUE_DEVICE(E, len);
  auto d_decomp = MAKE_UNIQUE_DEVICE(E, len);
  auto d_hist = MAKE_UNIQUE_DEVICE(F, bklen);
  auto h_hist = MAKE_UNIQUE_HOST(F, bklen);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  utils::fromfile(fname.c_str(), h_data.get(), len);
  memcpy_allkinds_async<H2D>(d_data.get(), h_data.get(), len, stream);
  cudaStreamSynchronize(stream);

  int grid_dim, block_dim, shmem_use, hist_repeat;
  psz::module::GPU_histogram_generic<E>::init(
      len, bklen, grid_dim, block_dim, shmem_use, hist_repeat);
  psz::module::GPU_histogram_generic<E>::kernel(
      d_data.get(), len, d_hist.get(), bklen, grid_dim, block_dim, shmem_use, hist_repeat, stream);
  memcpy_allkinds_async<D2H>(h_hist.get(), d_hist.get(), bklen, stream);
  cudaStreamSynchronize(stream);

  // auto buf = new phf::Buf<E>(len, bklen, -1, use_hfr);
  auto buf = std::make_unique<phf::Buf<E>>(len, bklen, -1, use_hfr);

  phf::high_level<E>::build_book(buf.get(), h_hist.get(), bklen, stream);

  uint8_t* d_encoded = nullptr;
  size_t encoded_len = 0;
  phf_header header{};

  double ms_enc = 1e9;
  for (int iter = 0; iter < repeat; ++iter) {
    cudaStreamSynchronize(stream);
    KernelTimer enc_timer;
    enc_timer.start(stream);
    if (use_hfr)
      phf::high_level<E>::encode_HFR(
          buf.get(), d_data.get(), len, &d_encoded, &encoded_len, header, stream);
    else
      phf::high_level<E>::encode(
          buf.get(), d_data.get(), len, &d_encoded, &encoded_len, header, stream);
    double this_enc = enc_timer.stop_ms(stream);
    if (this_enc < ms_enc) ms_enc = this_enc;
  }

  // decode timing (best-of-N kernel time)
  double ms_dec = 1e9;
  for (int iter = 0; iter < repeat; ++iter) {
    cudaStreamSynchronize(stream);
    KernelTimer dec_timer;
    dec_timer.start(stream);
    phf::high_level<E>::decode(buf.get(), header, d_encoded, d_decomp.get(), stream);
    double this_dec = dec_timer.stop_ms(stream);
    if (this_dec < ms_dec) ms_dec = this_dec;
  }

  auto identical = psz::module::GPU_identical(
      (void*)d_decomp.get(), (void*)d_data.get(), sizeof(E), len, stream);
  printf("\nUsing %s: ", use_hfr ? "HFR" : "HF");
  if (identical)
    printf("coding-decoding is successful.\n\n");
  else
    throw std::runtime_error("coding-decoding FAILED.\n");

  phf_print_header(
      &header, sizeof(E) == 1   ? "u1"
               : sizeof(E) == 2 ? "u2"
               : sizeof(E) == 4 ? "u4"
                                : "u8");
  print_GiBps<E>(len, ms_enc, use_hfr ? "encode-ReVISIT" : "encode-coarse");
  print_GiBps<u1>(encoded_len, ms_dec, "decode-coarse");
  printf("HF CR (in/out): %.2f\n", (double)(len * sizeof(E)) / encoded_len);

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  Arguments args;
  if (!args.parse(argc, argv)) return 1;

  if (args.use_cupti) {
    cuptiActivityRegisterCallbacks(phf_buf_req, phf_buf_done);
    g_cupti_active = true;
  }

  size_t len = args.total_len();

  if (args.type == "u1")
    hf_run<uint8_t>(args.fname, len, 256, args.use_hfr, args.repeat);
  else if (args.type == "u2")
    hf_run<uint16_t>(args.fname, len, args.bklen, args.use_hfr, args.repeat);
  else if (args.type == "u4")
    hf_run<uint32_t>(args.fname, len, args.bklen, args.use_hfr, args.repeat);
  else
    fprintf(stderr, "unknown type: %s\n", args.type.c_str());

  return 0;
}
