
#include <cuda_runtime.h>
#include <omp.h>

#include <vector>

// utilities for demo
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "cusz.h"
#include "cusz/type.h"
#include "utils/io.hh"  // from file
#include "utils/timer.hh"
using _portable::utils::fromfile;

using T = float;
using cor_t = psz_compressor*;
using copybuf_t = uint8_t*;
using header_t = psz_header*;

T *d_emu_ready_data, *h_emu_ready_data;

static const int rounds = 1000;

std::string fname;

// 1600 x1-segs (100 x16-segs), 825 MiB
static const size_t seg_len = (384 * 352 * 1);
static const size_t super_len = seg_len * 1600;
static const size_t oribytes = sizeof(T) * super_len;
static const size_t super_cyclic = 1600;

// 3D form
static const size_t x1_len = 384 * 352 * 1, x2_len = 384 * 352 * 2, x4_len = 384 * 352 * 4,
                    x8_len = 384 * 352 * 8, x16_len = 384 * 352 * 16;
static const psz_len3 x1_lrz1_len3 = {x1_len, 1, 1}, x2_lrz1_len3 = {x2_len, 1, 1},
                      x4_lrz1_len3 = {x4_len, 1, 1}, x8_lrz1_len3 = {x8_len, 1, 1},
                      x16_lrz1_len3 = {x16_len, 1, 1};
static const psz_len3 x1_lrz2_len3 = {384, 352, 1}, x2_lrz2_len3 = {384, 704, 1},
                      x4_lrz2_len3 = {768, 704, 1}, x8_lrz2_len3 = {768, 1408, 1},
                      x16_lrz2_len3 = {1536, 1408, 1};
static const psz_len3 x1_lrz3_len3 = {384, 352, 1}, x2_lrz3_len3 = {384, 352, 2},
                      x4_lrz3_len3 = {384, 352, 4}, x8_lrz3_len3 = {384, 352, 8},
                      x16_lrz3_len3 = {384, 352, 16};
// static const int x1_shift = 1, x2_shift = 2, x4_shift = 4, x8_shift = 8, x16_shift = 16;

auto mode = Abs;
auto eb = 3.f;

#define OMP_SCALE 1

void multistream(
    psz_len3 const in_len3, psz_predtype predictor = Lorenzo, psz_codectype codec = Huffman)
{
  auto exp_len = in_len3.x * in_len3.y * in_len3.z;

  printf("CPU proc. number:\t%d\n", omp_get_num_procs());
  printf("creating the same number of streams\n");

  auto streams = new cudaStream_t[omp_get_num_procs() * OMP_SCALE];
  auto cor_ins = new cor_t[omp_get_num_procs() * OMP_SCALE];
  auto copy_bufs = new copybuf_t[omp_get_num_procs() * OMP_SCALE];
  auto header_bufs = new header_t[omp_get_num_procs() * OMP_SCALE];

  // init data
  {
    cudaMalloc(&d_emu_ready_data, oribytes);
    cudaMallocHost(&h_emu_ready_data, oribytes);
    fromfile(fname, h_emu_ready_data, super_len);
    cudaMemcpy(d_emu_ready_data, h_emu_ready_data, oribytes, cudaMemcpyHostToDevice);
  }

// init OMP threads
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto nthread = omp_get_num_threads();
    printf("cpu_id @OMP scale=%d: %d/%d\n", OMP_SCALE, tid, nthread);

    for (auto i = 0; i < OMP_SCALE; i++) {
      auto ii = tid * OMP_SCALE + i;
      cudaStreamCreate(&streams[ii]);
      cor_ins[ii] = psz_create(F4, in_len3, predictor, 64, codec);
      copy_bufs[ii] = new uint8_t[exp_len * 2];  // CR upperlim=2
      header_bufs[ii] = new psz_header;
    }
  }

  auto a = hires::now();

  uint8_t* d_compressed;
  size_t comp_len;
  void* comp_timerecord;
  auto dummy_copy_len = 1 / 4 * exp_len * sizeof(f4);
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();

    for (auto i = 0; i < OMP_SCALE; i++) {
      auto ii = tid * OMP_SCALE + i;

      for (auto r = 0; r < rounds; r++) {
        auto this_tid_shift = r * (omp_get_num_procs() * OMP_SCALE * exp_len) +
                              tid * OMP_SCALE * exp_len + ii * exp_len;
        this_tid_shift = this_tid_shift % super_cyclic;

        psz_compress(
            cor_ins[ii], d_emu_ready_data + this_tid_shift, in_len3, eb, mode, &d_compressed,
            &comp_len, header_bufs[ii], nullptr, streams[ii]);
        cudaMemcpyAsync(
            copy_bufs[ii], d_compressed,  //
            /* ERROR when using comp_len */ dummy_copy_len, cudaMemcpyDeviceToDevice, streams[ii]);
        // cudaStreamSynchronize(streams[ii]);

        psz_clear_buffer(cor_ins[ii]);  // double check if it affects d_compressed buffer
      }
      cudaStreamSynchronize(streams[ii]);
    }
  }
  auto b = hires::now();

  auto bytes_processed = rounds * exp_len * omp_get_num_procs() * sizeof(f4);
  auto seconds = static_cast<duration_t>(b - a).count();

  const auto GiB = 1024 * 1024 * 1024.0;

  printf("total time: %f\n", seconds);
  printf("bytes processed: %lu\n", bytes_processed);
  printf("speed: %.2f GiB/s\n", bytes_processed / GiB / seconds);

// clean up
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();

    psz_release(cor_ins[tid]);
    cudaStreamDestroy(streams[tid]);

    delete[] copy_bufs[tid];
    delete header_bufs[tid];
  }
  delete[] streams;
  delete[] copy_bufs;
  delete[] header_bufs;
  delete[] cor_ins;
}

int main(int argc, char** argv)
{
  auto help = []() { printf("PROG </path/to/emu-data> <1. normal; 2. FZG> <seg:1,2,4,8,16>\n"); };

  if (argc < 3) {
    help();
    exit(0);
  }

  fname = std::string(argv[1]);
  int method = std::stoi(argv[2]);
  int seg = std::stoi(argv[3]);

  psz_len3 exp_len3;

  if (seg == 1)
    exp_len3 = x1_lrz1_len3;
  else if (seg == 2)
    exp_len3 = x2_lrz1_len3;
  else if (seg == 4)
    exp_len3 = x4_lrz1_len3;
  else if (seg == 8)
    exp_len3 = x8_lrz1_len3;
  else if (seg == 16)
    exp_len3 = x16_lrz1_len3;
  else {
    help();
    exit(0);
  }

  if (method == 1)
    multistream(exp_len3, Lorenzo, Huffman);
  else if (method == 2)
    multistream(exp_len3, LorenzoZigZag, FZGPUCodec);
  else
    help();

  return 0;
}