#include <cuda_runtime.h>

#include <bitset>
#include <iostream>

#include "compbuf.hh"
#include "cxx_hfbk.h"
#include "hfcxx_module.hh"
#include "kernel/lrz/lrz.gpu.hh"
#include "kernel/spv.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"
#include "utils/viewer.hh"

using std::cout;
using std::endl;

using _portable::utils::fromfile;

using Buf = psz::CompressorBuffer<float>;
using BufToggle = psz::CompressorBufferToggle;
using T = float;
using E = Buf::E;
using M = Buf::M;
using Hf = Buf::Hf;
int const radius = 64;
double eb = 3.0;
double eb_r, ebx2, ebx2_r;

GPU_unique_hptr<T[]> h_uncomp;
GPU_unique_dptr<T[]> d_uncomp;
GPU_unique_hptr<T[]> h_decomp;
GPU_unique_dptr<T[]> d_decomp;

std::array<size_t, 3> len3_std;

Buf* mem;
cudaStream_t stream;

void run(char* fname, size_t const len)
{
  fromfile(fname, h_uncomp.get(), len);
  memcpy_allkinds<H2D>(d_uncomp.get(), h_uncomp.get(), len);

  psz::module::GPU_c_lorenzo_nd_with_outlier<T, false, E, /* uncomp-place enc */ true>(
      d_uncomp.get(), len3_std, mem->ectrl(), (void*)mem->outlier(), mem->top1(), ebx2, ebx2_r,
      radius,
      stream,  //
      mem->pbk(), mem->pbk_res_tree_IDs(), mem->pbk_res_bitstream(), mem->pbk_res_bits(),
      mem->pbk_res_entries(), mem->pbk_res_loc());

  cudaStreamSynchronize(stream);
  mem->pbk_encoding_summary();

  auto splen = mem->compact->num_outliers();
  auto endloc = mem->pbk_encoding_endloc();

  printf("splen: %d\n", splen);

  {
    // clang-format off
    memcpy_allkinds<D2H>(mem->h_pbk_res_tree_IDs.get(), mem->d_pbk_res_tree_IDs.get(), mem->num_chunk);
    memcpy_allkinds<D2H>(mem->h_pbk_res_entries.get(), mem->d_pbk_res_entries.get(), mem->num_chunk);
    memcpy_allkinds<D2H>(mem->h_pbk_res_bits.get(), mem->d_pbk_res_bits.get(), mem->num_chunk);
    memcpy_allkinds<D2H>(mem->h_pbk_res_bitstream.get(), mem->d_pbk_res_bitstream.get(), endloc);
    // clang-format on

    // for (auto i = 0; i < 100; i++) {
    //   auto val = mem->h_pbk_res_bitstream[i];
    //   cout << i << "\t" << val << "\t" << std::bitset<32>(val) << endl;
    // }
  }

  float _;
  if (splen)
    psz::spv_scatter_naive<CUDA, T, M>(
        mem->compact_val(), mem->compact_idx(), splen, d_decomp.get(), &_, stream);
  cudaStreamSynchronize(stream);

  // phf::cuhip::modules<E, Hf>::GPU_pbk_coarse_decode(
  //     mem->pbk_res_bitstream(), endloc, mem->pbk_revbooks_11(), mem->PBK_REVBK_BYTES,
  //     mem->pbk_res_tree_IDs(), mem->pbk_res_bits(), mem->pbk_res_entries(), mem->ectrl(), len,
  //     stream);
  // cudaStreamSynchronize(stream);

  {
    auto h_ectrl = MAKE_UNIQUE_HOST(E, len);
    phf::cuhip::modules<E, Hf>::CPU_pbk_coarse_decode(
        mem->pbk_res_bitstream_h(), endloc, mem->pbk_revbooks_11_h(), mem->PBK_REVBK_BYTES,
        mem->pbk_res_tree_IDs_h(), mem->pbk_res_bits_h(), mem->pbk_res_entries_h(), h_ectrl.get(),
        len);
    memcpy_allkinds<H2D>(mem->ectrl(), h_ectrl.get(), len);

    // for (auto j = 0; j < 352; j++) {
    //   printf("y=%03d\t", j);
    //   for (auto i = 0; i < 384; i++) {  //
    //     printf("%3d ", (int)h_ectrl[j * 384 + i]);
    //   }
    //   printf("\n");
    // }
  }

  auto d_space = d_decomp.get(), d_xdata = d_decomp.get();  // aliases
  psz::module::GPU_x_lorenzo_nd<T, false, E>(
      mem->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, radius, stream);
  cudaStreamSynchronize(stream);

  psz::analysis::GPU_evaluate_quality_and_print(
      d_decomp.get(), d_uncomp.get(), len, mem->pbk_bytes);

  memcpy_allkinds<D2H>(h_decomp.get(), d_decomp.get(), len);

  // for (auto j = 0; j < 352; j++) {
  //   printf("y=%03d\t", j);
  //   for (auto i = 0; i < 384; i++) {  //
  //     printf("%3d ", (int)h_decomp[j * 384 + i]);
  //   }
  //   printf("\n");
  // }
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    printf("       0       1      2  3  4\n");
    printf("usage: pbk_run fname  x  y  z\n");
    return 1;
  }

  size_t x = atoi(argv[2]), y = atoi(argv[3]), z = atoi(argv[4]);
  BufToggle toggle{
      .err_ctrl_quant = true,
      .compact_outlier = true,
      .anchor = true,
      .pbk_all = true,
  };
  mem = new Buf(x, y, z, 64, false, &toggle);
  size_t len = x * y * z;
  len3_std = MAKE_STDLEN3(x, y, z);
  h_uncomp = MAKE_UNIQUE_HOST(T, len);
  d_uncomp = MAKE_UNIQUE_DEVICE(T, len);
  h_decomp = MAKE_UNIQUE_HOST(T, len);
  d_decomp = MAKE_UNIQUE_DEVICE(T, len);
  cudaStreamCreate(&stream);
  eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  // // check books
  // for (auto i = 0; i < mem->PBK_N; i++) {
  //   cout << "book/tree id:\t" << i << endl;
  //   phf_book_view(mem->h_pbk_r64.get() + i * mem->PBK_LEN, mem->PBK_LEN);
  //   cout << "--------------------" << endl;
  // }

  run(argv[1], len);

  delete mem;
  cudaStreamDestroy(stream);

  return 0;
}