#include <cuda_runtime.h>

#include <bitset>
#include <cmath>
#include <iostream>

#include "compbuf.hh"
#include "cxx_hfbk.h"
#include "ex_utils2.hh"
#include "hfcxx_module.hh"
#include "kernel/lrz/lrz.gpu.hh"
#include "kernel/spv.hh"
#include "mem/cxx_backends.h"
#include "stat/compare.hh"
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
const u2 fixed_radius = 64;
double eb = 3.0;
double eb_r, ebx2, ebx2_r;
const auto width = 5;

GPU_unique_hptr<T[]> h_uncomp;
GPU_unique_dptr<T[]> d_uncomp;
GPU_unique_hptr<T[]> h_decomp_eip;
GPU_unique_dptr<T[]> d_decomp_eip;
GPU_unique_hptr<T[]> h_decomp_ref;
GPU_unique_dptr<T[]> d_decomp_ref;

GPU_unique_hptr<E[]> h_ectrl_eip;
GPU_unique_hptr<E[]> h_ectrl_ref;

std::array<size_t, 3> len3_std;

Buf* mem_eip;
Buf* mem_ref;
cudaStream_t stream;

#define PRINT_ALL_11_TREES()                                              \
  for (auto i = 0; i < mem->PBK_N; i++) {                                 \
    cout << "book/tree id:\t" << i << endl;                               \
    phf_book_view(mem->h_pbk_r64.get() + i * mem->PBK_LEN, mem->PBK_LEN); \
    cout << "--------------------" << endl;                               \
  }

#define PRINT_H_BITSTREAM(N)                                          \
  for (auto i = 0; i < N; i++) {                                      \
    auto val = mem->h_pbk_res_bitstream[i];                           \
    cout << i << "\t" << val << "\t" << std::bitset<32>(val) << endl; \
  }

#define PRINT_ALL_H_ECTRL()                                                                \
  for (auto j = 0; j < 352; j++) {                                                         \
    printf("y=%03d\t", j);                                                                 \
    for (auto i = 0; i < 384; i++) { printf("%3d ", (int)h_ectrl_eip[j * 384 + i] - 64); } \
    printf("\n");                                                                          \
  }

#define PRINT_ALL_H_DECOMP()                                                           \
  memcpy_allkinds<D2H>(h_decomp_eip.get(), d_decomp_eip.get(), len);                   \
  for (auto j = 0; j < 352; j++) {                                                     \
    printf("y=%03d\t", j);                                                             \
    for (auto i = 0; i < 384; i++) { printf("%3d ", (int)h_decomp_eip[j * 384 + i]); } \
    printf("\n");                                                                      \
  }

#define EIP_DECODE_GPU()                                                                          \
  phf::cuhip::modules<E, Hf>::GPU_pbk_coarse_decode(                                              \
      mem_eip->pbk_res_bitstream(), endloc, mem_eip->pbk_revbooks_11(), mem_eip->PBK_REVBK_BYTES, \
      mem_eip->pbk_res_tree_IDs(), mem_eip->pbk_res_bits(), mem_eip->pbk_res_entries(),           \
      mem_eip->ectrl(), len, stream);                                                             \
  cudaStreamSynchronize(stream);

#define EIP_DECODE_CPU()                                                                  \
  phf::cuhip::modules<E, Hf>::CPU_pbk_coarse_decode(                                      \
      mem_eip->pbk_res_bitstream_h(), endloc, mem_eip->pbk_revbooks_11_h(),               \
      mem_eip->PBK_REVBK_BYTES, mem_eip->pbk_res_tree_IDs_h(), mem_eip->pbk_res_bits_h(), \
      mem_eip->pbk_res_entries_h(), h_ectrl_eip.get(), len);

#define PRINT_STATS(type, color, radius, cr) \
  printf(                                    \
      type                                   \
      "\t"                                   \
      "R\t" color                            \
      "%u\e[0m\t"                            \
      "CR\t" color                           \
      "%.2lf\e[0m\t"                         \
      "PSNR\t" color                         \
      "%.2lf\e[0m\t"                         \
      "NRMSE\t" color                        \
      "%lf\e[0m\t"                           \
      "MAX.ABS.EB\t" color                   \
      "%lf\e[0m\t"                           \
      "MAX.REL.EB\t" color                   \
      "%lf\e[0m\t"                           \
      "T.SPLEN\t" color "%d\e[0m\t",         \
      radius, cr, s->score_PSNR, s->score_NRMSE, s->max_err_abs, s->max_err_rel, splen)

void driver_program(const char* fname, size_t const len, bool EIP_verbose = false)
{
  fromfile(fname, h_uncomp.get(), len);
  memcpy_allkinds<H2D>(d_uncomp.get(), h_uncomp.get(), len);

  auto reference_run_low_level = [&](u2 radius = fixed_radius) {
    psz::module::GPU_c_lorenzo_nd_with_outlier<T, false, E, /* uncomp-place enc */ false>(
        d_uncomp.get(), len3_std, mem_ref->ectrl(), (void*)mem_ref->outlier(), mem_ref->top1(),
        ebx2, ebx2_r, radius, stream);
    cudaStreamSynchronize(stream);

    auto splen = mem_ref->compact->num_outliers();

    if (splen)
      psz::spv_scatter_naive<CUDA, T, M>(
          mem_ref->compact_val(), mem_ref->compact_idx(), splen, d_decomp_ref.get(), nullptr,
          stream);
    cudaStreamSynchronize(stream);

    memcpy_allkinds<D2H>(h_ectrl_ref.get(), mem_ref->ectrl(), len);
    // PRINT_ALL_H_ECTRL();

    auto d_space = d_decomp_ref.get(), d_xdata = d_decomp_ref.get();  // aliases
    psz::module::GPU_x_lorenzo_nd<T, false, E>(
        mem_ref->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, radius, stream);
    cudaStreamSynchronize(stream);

    auto s = new psz_statistics;
    psz::cuhip::GPU_assess_quality(s, d_uncomp.get(), d_decomp_ref.get(), len);
    PRINT_STATS("modular ref.", "\e[34m", radius, NAN);
    printf("\n");

    // PRINT_ALL_H_DECOMP();

    // reset
    memset_device(mem_ref->compact->d_num.get(), 1);
    memset_device(d_decomp_ref.get(), len);
  };

  auto EncodeInPlace_run = [&]() {
    psz::module::GPU_c_lorenzo_nd_with_outlier<T, false, E, /* uncomp-place enc */ true>(
        d_uncomp.get(), len3_std, mem_eip->ectrl(), (void*)mem_eip->outlier(), mem_eip->top1(),
        ebx2, ebx2_r, fixed_radius, stream,  //
        mem_eip->pbk(), mem_eip->pbk_res_tree_IDs(), mem_eip->pbk_res_bitstream(),
        mem_eip->pbk_res_bits(), mem_eip->pbk_res_entries(), mem_eip->pbk_res_loc(),  //
        mem_eip->d_pbk_brval.get(), mem_eip->d_pbk_bridx.get(), mem_eip->d_pbk_brnum.get());
    cudaStreamSynchronize(stream);

    mem_eip->pbk_encoding_summary(EIP_verbose);

    auto splen = mem_eip->compact->num_outliers();
    auto endloc = mem_eip->pbk_encoding_endloc();

    // clang-format off
    memcpy_allkinds<D2H>(mem_eip->h_pbk_res_tree_IDs.get(), mem_eip->d_pbk_res_tree_IDs.get(), mem_eip->num_chunk);
    memcpy_allkinds<D2H>(mem_eip->h_pbk_res_entries.get(), mem_eip->d_pbk_res_entries.get(), mem_eip->num_chunk);
    memcpy_allkinds<D2H>(mem_eip->h_pbk_res_bits.get(), mem_eip->d_pbk_res_bits.get(), mem_eip->num_chunk);
    memcpy_allkinds<D2H>(mem_eip->h_pbk_res_bitstream.get(), mem_eip->d_pbk_res_bitstream.get(), endloc);
    // clang-format on

    // !!!! TODO implicitly done in summary
    // memcpy_allkinds<D2H>(mem_eip->h_pbk_brnum.get(), mem_eip->d_pbk_brnum.get(), 1);
    auto brnum = mem_eip->h_pbk_brnum[0];
    // if (brnum) printf("There are artificial top symbols.\n");

    // PRINT_H_BITSTREAM(100);

    if (splen)
      psz::spv_scatter_naive<CUDA, T, M>(
          mem_eip->compact_val(), mem_eip->compact_idx(), splen, d_decomp_eip.get(), nullptr,
          stream);
    cudaStreamSynchronize(stream);

    EIP_DECODE_CPU();
    // PRINT_ALL_H_ECTRL();

    memcpy_allkinds<H2D>(mem_eip->ectrl(), h_ectrl_eip.get(), len);

    // fix ectrl from artificial top symbols
    if (brnum)
      psz::spv_scatter_naive<CUDA, E, M>(
          mem_eip->d_pbk_brval.get(), mem_eip->d_pbk_bridx.get(), brnum, mem_eip->ectrl(), nullptr,
          stream);
    cudaStreamSynchronize(stream);

    auto d_space = d_decomp_eip.get(), d_xdata = d_decomp_eip.get();  // aliases
    psz::module::GPU_x_lorenzo_nd<T, false, E>(
        mem_eip->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, fixed_radius, stream);
    cudaStreamSynchronize(stream);

    auto s = new psz_statistics;
    psz::cuhip::GPU_assess_quality(s, d_uncomp.get(), d_decomp_eip.get(), len);
    PRINT_STATS(
        "EIP \e[33m(PBK HF)\e[0m", "\e[33m", fixed_radius,
        len * sizeof(T) * 1.0 / mem_eip->pbk_bytes);
    if (brnum)
      printf("E/BR.SPLEN\t\e[33m%u\e[0m\n", brnum);
    else
      printf("\n");

    // PRINT_ALL_H_DECOMP();

    // reset status
    memset_device(mem_eip->compact->d_num.get(), 1);
    memset_device(mem_eip->d_pbk_brnum.get(), 1);
    memset_device(mem_eip->d_pbk_res_loc.get(), 1);
    memset_device(d_decomp_eip.get(), len);
  };

  EncodeInPlace_run();
  reference_run_low_level();
}

int main(int argc, char** argv)
{
  Arguments args = parse_arguments(argc, argv);
  const size_t len = args.x * args.y * args.z;

  size_t x = atoi(argv[2]), y = atoi(argv[3]), z = atoi(argv[4]);
  BufToggle toggle_eip{.pbk_all = true}, toggle_ref{};

  try {
    mem_eip = new Buf(args.x, args.y, args.z, false, &toggle_eip);
    mem_ref = new Buf(args.x, args.y, args.z, false, &toggle_ref);
  }
  catch (const std::logic_error& e) {
    std::cerr << e.what() << ".\n";
    std::cerr << "Check if shell variables PBK_BOOK and PBK_RVBK are set" << std::endl;
    return 1;
  }

  len3_std = MAKE_STDLEN3(args.x, args.y, args.z);

  h_uncomp = MAKE_UNIQUE_HOST(T, len);
  d_uncomp = MAKE_UNIQUE_DEVICE(T, len);

  h_decomp_eip = MAKE_UNIQUE_HOST(T, len);
  h_decomp_ref = MAKE_UNIQUE_HOST(T, len);
  d_decomp_eip = MAKE_UNIQUE_DEVICE(T, len);
  d_decomp_ref = MAKE_UNIQUE_DEVICE(T, len);
  h_ectrl_eip = MAKE_UNIQUE_HOST(E, len);
  h_ectrl_ref = MAKE_UNIQUE_HOST(E, len);

  cudaStreamCreate(&stream);
  eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  // check books
  // PRINT_ALL_11_TREES();

  auto file_names = construct_file_names(
      args.fname_prefix, args.fname_suffix, args.from_number, args.to_number, width);

  for (const auto& fname : file_names) {
    cout << "\n-------------------- " << fname << " --------------------" << endl;
    driver_program(fname.c_str(), len, args.verbose);
  }

  delete mem_eip;
  delete mem_ref;
  cudaStreamDestroy(stream);

  return 0;
}