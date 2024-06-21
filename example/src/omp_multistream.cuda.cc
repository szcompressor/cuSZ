/**
 * @file demo_capi.cuda.cc
 * @author Jiannan Tian
 * @brief Also see demo_capi_minimal.cc for a more concise view.
 * @version 0.10
 * @date 2022-05-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include <omp.h>

#include <vector>

#include "cusz.h"

// utilities for demo
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "cusz/review.h"
#include "cusz/type.h"
#include "utils/io.hh"  // io::read_binary_to_array

using T = float;
using compressor_t = psz_compressor*;
using copybuf_t = uint8_t*;
using header_t = psz_header*;

T *d_emu_ready_data, *h_emu_ready_data;

static const int rounds = 1000;

std::string fname;
static const size_t super_len = 384 * 352 * 1600;  // 150 snapshots
static const size_t oribytes = sizeof(T) * super_len;
static const size_t super_cyclic = 1600;

static const size_t unit_len_4 = 384 * 352 * 4;
static const psz_len3 unit_len3_4 = {384, 352, 4};
static const int shift_4 = 4;

static const size_t unit_len_6 = 384 * 352 * 6;
static const psz_len3 unit_len3_6 = {384, 352, 6};
static const int shift_6 = 6;

static const size_t unit_len_8 = 384 * 352 * 8;
static const psz_len3 unit_len3_8 = {384, 352, 8};
static const int shift_8 = 8;

static const size_t selected_len = unit_len_6;
static const psz_len3 selected_len3 = unit_len3_6;
static const int selected_shift = shift_6;

auto mode = Abs;  // set compression mode
auto eb = 3;      // set error bound

void multistream(psz_predtype predictor, psz_len3 const in_len3)
{
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("creating the same number of streams\n");

  auto streams = new cudaStream_t[omp_get_num_procs()];
  auto compressor_instances = new compressor_t[omp_get_num_procs()];
  auto copy_buffers = new copybuf_t[omp_get_num_procs()];
  auto header_buffers = new header_t[omp_get_num_procs()];

  // initialize ready data

  {
    cudaMalloc(&d_emu_ready_data, oribytes);
    cudaMallocHost(&h_emu_ready_data, oribytes);
    io::read_binary_to_array(fname, h_emu_ready_data, super_len);
    cudaMemcpy(
        d_emu_ready_data, h_emu_ready_data, oribytes, cudaMemcpyHostToDevice);
  }

// initialize each thread
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto num_cpu_threads = omp_get_num_threads();

    printf("cpu_id: %d/%d\n", tid, num_cpu_threads);

    // clang-format off
    cudaStreamCreate(&streams[tid]);
    compressor_instances[tid] = psz_create(F4, in_len3, predictor, 512, Huffman);
    copy_buffers[tid] = new uint8_t[selected_len * 2];  // CR upperlim=2
    header_buffers[tid] = new psz_header;
    // clang-format on
  }

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();

    uint8_t* p_compressed;
    size_t comp_len;
    void* comp_timerecord;

    auto dummy_copy_len = 1 / 4 * selected_len * sizeof(f4);

    for (auto i = 0; i < rounds; i++) {
      auto this_tid_shift =
          i * (omp_get_num_procs() * selected_len) + tid * selected_len;
      this_tid_shift = this_tid_shift % super_cyclic;

      psz_compress(
          compressor_instances[tid], d_emu_ready_data + this_tid_shift,
          in_len3, eb, mode, &p_compressed, &comp_len, header_buffers[tid],
          comp_timerecord, streams[tid]);
      cudaMemcpyAsync(
          copy_buffers[tid], p_compressed, dummy_copy_len,
          cudaMemcpyDeviceToDevice, streams[tid]);
    }
    cudaStreamSynchronize(streams[tid]);
  }

// clean up
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();

    psz_release(compressor_instances[tid]);
    cudaStreamDestroy(streams[tid]);

    delete[] copy_buffers[tid];
    delete header_buffers[tid];
  }
  delete[] streams;
  delete[] copy_buffers;
  delete[] header_buffers;
  delete[] compressor_instances;
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    /* For demo, we use 3600x1800 CESM data. */
    printf("PROG /path/to/super-data\n");
    exit(0);
  }

  fname = std::string(argv[1]);

  multistream(Lorenzo, selected_len3);

  return 0;
}