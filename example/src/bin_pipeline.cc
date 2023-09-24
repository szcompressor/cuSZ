/**
 * @file bin_pipeline.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "ex_utils.hh"
#include "hf/hf.hh"
#include "kernel.hh"
#include "mem.hh"
#include "port.hh"
#include "stat.hh"
// #include "utils.hh"
#include "utils/viewer.hh"

#define ABS 0
#define REL 1

#define LORENZO 11
#define SPLINE3 12

int g_splen;

#define PRINT_STATUS                                              \
  {                                                               \
    printf("\033[1mfname=\033[0m\"%s\"\n", ifn);                  \
    printf("\033[1m(x, y, z)=\033[0m(%lu, %lu, %lu)\n", x, y, z); \
    printf(                                                       \
        "\033[1m(eb, mode, radius)=\033[0m(%lf, %s, %d)\n", eb,   \
        mode == REL ? "REL" : "ABS", radius);                     \
    printf(                                                       \
        "\033[1mpredictor=\033[0m%s\n",                           \
        Predictor == SPLINE3 ? "spline3" : "lorenzo");            \
    if (mode == REL) printf("\033[1meb_rel=\033[0m%lf\n", eb);    \
  }

string dtype, etype, mode, pred, eb_str;
f8 _1, _2, rng;

namespace {

szt tune_coarse_huffman_sublen(szt len)
{
  int current_dev = 0;
  GpuSetDevice(current_dev);
  GpuDeviceProp dev_prop{};
  GpuGetDeviceProperties(&dev_prop, current_dev);

  // auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread =
      allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
  auto optimal_sublen = psz_utils::get_npart(len, deflate_nthread);
  optimal_sublen =
      psz_utils::get_npart(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) *
      HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
}

int ndim(dim3 l)
{
  auto n = 3;
  if (l.z == 1) n = 2;
  if (l.y == 1) n = 1;
  return n;
}

void radius_legal(int const radius, int const sizeof_T)
{
  size_t upper_bound = 1lu << (sizeof_T * 8);
  // cout << upper_bound << endl;
  // cout << radius * 2 << endl;
  if ((radius * 2) > upper_bound)
    throw std::runtime_error("Radius overflows error-quantization type.");
}

string suffix(bool compat = false)
{
  if (not compat)
    return pred + "_" + string(dtype == "f" ? "f4" : "f8") + etype + "_" +
           mode + eb_str;
  else
    return pred + "_" + string(dtype == "f" ? "f4" : "f8") + "u4" + "_" +
           mode + eb_str;
}

}  // namespace

#define MemPool pszmempool_cxx<T, E, H>

void print_tobediscarded_info(float time_in_ms, string fn_name)
{
  // auto title = "[psz::info::discard::" + fn_name + "]";
  // printf("%s time (ms): %.6f\n", title.c_str(), time_in_ms);
}

template <typename T>
float print_GBps(szt len, float time_in_ms, string fn_name)
{
  auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto GiBps = len * sizeof(T) * 1.0 / B_to_GiB / (time_in_ms / 1000);
  auto title = "[psz::info::res::" + fn_name + "]";
  printf(
      "%s shortest time (ms): %.6f\thighest throughput (GiB/s): %.2f\n",
      title.c_str(), time_in_ms, GiBps);
  return GiBps;
}

template <int Predictor, typename T = f4, typename E = u4, typename H = u4>
void demo_c_predict(
    MemPool* mem, pszmem_cxx<u4>* ectrl_u4, char const* ifn, f8 const eb,
    int const radius, GpuStreamT stream, bool proto = false)
{
  using FP = T;

  auto len = mem->od->len();
  auto len3 = mem->el->template len3<dim3>();
  auto len3p = mem->es->template len3<dim3>();
  f4 time, time_histcpu_base, time_histcpu_optim;

  // psz_comp_l23<T, E, FP>(mem->od->dptr(), len3, eb, radius,
  // mem->ectrl_lrz(), mem->outlier_space(), &time, stream);

  // ----------------------------------------
  auto time_pred = (float)INT_MAX;
  for (auto i = 0; i < 10; i++) {
    psz_comp_l23r<T, E>(
        mem->od->dptr(), len3, eb, radius, mem->ectrl_lrz(),
        (void*)mem->compact, &time, stream);
    print_tobediscarded_info(time, "comp_pred_l23r");
    time_pred = std::min(time, time_pred);
  }
  print_GBps<T>(len, time_pred, "comp_pred");

  mem->compact->make_host_accessible((GpuStreamT)stream);
  g_splen = mem->compact->num_outliers();

  cout << "[psz::info] outlier: " << g_splen << endl;

  mem->el
      ->control({D2H})  //
      ->file(string(string(ifn) + ".eq." + suffix()).c_str(), ToFile);
  mem->el->castto(ectrl_u4, psz_space::Host)->control({H2D});
}

template <int Predictor, typename T = f4, typename E = u4, typename H = u4>
void demo_d_predict(
    MemPool* mem, double const eb, int const radius, GpuStreamT stream,
    bool proto = false)
{
  using FP = T;

  auto len = mem->el->len();
  auto len3 = mem->el->template len3<dim3>();
  auto len3p = mem->es->template len3<dim3>();
  f4 time_pred, time_scatter;

  // ----------------------------------------
  auto time_scatter_min = (float)INT_MAX;
  for (auto i = 0; i < 10; i++) {
    psz::spv_scatter<PROPER_GPU_BACKEND, T, u4>(
        mem->compact_val(), mem->compact_idx(), g_splen, mem->outlier_space(),
        &time_scatter, stream);

    print_tobediscarded_info(time_scatter, "decomp_scatter");
    time_scatter_min = std::min(time_scatter, time_scatter_min);
  }
  print_GBps<T>(len, time_scatter_min, "decomp_scatter");

  // ----------------------------------------
  auto time_pred_min = (float)INT_MAX;
  for (auto i = 0; i < 10; i++) {
    psz_decomp_l23<T, E, FP>(
        mem->ectrl_lrz(), len3, mem->outlier_space(), eb, radius,
        mem->xd->dptr(), &time_pred, stream);

    print_tobediscarded_info(time_pred, "decomp_scatter");
    time_pred_min = std::min(time_pred, time_pred_min);
  }
  print_GBps<T>(len, time_pred_min, "decomp_pred");
}

template <typename H>
void demo_hist_u4in(
    pszmem_cxx<u4>* hist_in, pszmem_cxx<H>* hist_out, char const* ifn,
    int const radius, GpuStreamT stream)
{
  using E = u4;

  f4 tgpu_base, tgpu_optim;
  f4 tcpu_base, tcpu_optim;

  auto ser_base = new pszmem_cxx<u4>(radius * 2, 1, 1, "hist-normal");
  auto ser_optim = new pszmem_cxx<E>(radius * 2, 1, 1, "hist-sp_cpu");
  auto gpu_optim = new pszmem_cxx<E>(radius * 2, 1, 1, "hist-sp_gpu");
  ser_base->control({MallocHost});
  ser_optim->control({MallocHost});
  gpu_optim->control({MallocHost, Malloc});

  auto bklen = radius * 2;
  hist<SEQ, E>(
      false, hist_in->hptr(), hist_in->len(), ser_base->hptr(), bklen,
      &tcpu_base, stream);
  hist<SEQ, E>(
      true, hist_in->hptr(), hist_in->len(), ser_optim->hptr(), bklen,
      &tcpu_optim, stream);

  // ----------------------------------------
  auto time_hist_min = (float)INT_MAX;
  for (auto i = 0; i < 10; i++) {
    hist<PROPER_GPU_BACKEND, E>(
        false, hist_in->dptr(), hist_in->len(), hist_out->dptr(), bklen,
        &tgpu_base, stream);

    print_tobediscarded_info(tgpu_base, "comp_hist");
    time_hist_min = std::min(time_hist_min, tgpu_base);
  }
  // [psz::caveat] hard coded throughput calculating
  print_GBps<f4>(hist_in->len(), time_hist_min, "comp_hist");

#if defined(PSZ_USE_CUDA)
  hist<PROPER_GPU_BACKEND, E>(
      true, hist_in->dptr(), hist_in->len(), gpu_optim->dptr(), bklen,
      &tgpu_optim, stream);
#elif defined(PSZ_USE_HIP)
  cout << "[psz::warning] skip hist-optim (hang when specifying HIP)" << endl;
#endif

  ser_base->file(string(string(ifn) + ".ht." + suffix()).c_str(), ToFile);

#if !defined(PSZ_USE_HIP)
  // [psz::warning] The HIP testbed does not have high-frequency CPU. Also,
  // hist_sp does not work.
  printf(
      "hist cpu baseline:\t%5.2f ms\toptim:\t%5.2f ms (speedup: %3.2fx)\n",  //
      tcpu_base, tcpu_optim, tcpu_base / tcpu_optim);
  printf(
      "hist gpu baseline:\t%5.2f ms\toptim:\t%5.2f ms (speedup: %3.2fx)\n",  //
      tgpu_base, tgpu_optim, tgpu_base / tgpu_optim);
#endif

  delete gpu_optim;
  delete ser_base;
  delete ser_optim;
}

template <
    int Predictor = LORENZO, typename T = f4, typename E = u4, typename H = u4>
void demo_pipeline(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    f8 eb = 1.2e-4, int mode = REL, int const radius = 32, bool proto = false)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;
  using M = u4;

  radius_legal(radius, sizeof(E));

  auto len = x * y * z;
  auto len3 = dim3(x, y, z);
  f4 time, time_histcpu_base, time_histcpu_optim;

  auto sublen = tune_coarse_huffman_sublen(len);
  auto pardeg = psz_utils::get_npart(len, sublen);

  cusz::HuffmanCodec<u4, u4> hf_codec;

  auto mem = new MemPool(x, radius, y, z);

  mem->od->control({Malloc, MallocHost})
      ->file(ifn, FromFile)
      ->control({H2D})
      ->extrema_scan(_1, _2, rng);
  mem->xd->control({Malloc, MallocHost});

  auto ectrl_u4 = new pszmem_cxx<u4>(x, y, z, "ectrl_u4");
  auto ectrl_u4_decoded = new pszmem_cxx<u4>(x, y, z, "ectrl_u4_dec");

  ectrl_u4->control({Malloc, MallocHost});
  ectrl_u4_decoded->control({Malloc, MallocHost});

  hf_codec.init(
      ectrl_u4->len(), radius * 2, pardeg /* not optimal for perf */);

  GpuStreamT stream;
  GpuStreamCreate(&stream);

  if (mode == REL) eb *= rng;

  PRINT_STATUS;

  demo_c_predict<Predictor, T, E, H>(
      mem, ectrl_u4, ifn, eb, radius, stream, proto);

  auto num_outlier =
      Predictor == SPLINE3
          ? count_outlier<E>(
                mem->es->dptr(), mem->es->len(), radius, (void*)stream)
          : count_outlier<E>(
                mem->el->dptr(), mem->el->len(), radius, (void*)stream);
  // printf("#outlier:\t%u\n", num_outlier);

  demo_hist_u4in<H>(ectrl_u4, mem->ht, ifn, radius, stream);

  size_t hf_outlen;
  {  // Huffman
    auto hf_inlen = ectrl_u4->len();
    u1* d_encoded;

    // hf_codec.build_codebook(mem->hist(), radius * 2, stream);
    hf_codec.build_codebook(mem->ht, radius * 2, stream);

    // ----------------------------------------
    auto time_comp_lossless = (float)INT_MAX;
    for (auto i = 0; i < 10; i++) {
      hf_codec.encode(
          ectrl_u4->dptr(), ectrl_u4->len(), &d_encoded, &hf_outlen, stream);

      print_tobediscarded_info(hf_codec.time_lossless(), "comp_hf_encode");
      time_comp_lossless =
          std::min(time_comp_lossless, hf_codec.time_lossless());
    }
    print_GBps<f4>(len, time_comp_lossless, "comp_hf_encode");

    // ----------------------------------------
    auto time_decomp_lossless = (float)INT_MAX;
    for (auto i = 0; i < 10; i++) {
      hf_codec.decode(d_encoded, ectrl_u4_decoded->dptr());

      print_tobediscarded_info(hf_codec.time_lossless(), "decomp_hf_decode");
      time_comp_lossless =
          std::min(time_comp_lossless, hf_codec.time_lossless());
    }
    print_GBps<f4>(len, time_comp_lossless, "decomp_hf_decode");

    auto identical = psz::thrustgpu_identical(
        ectrl_u4_decoded->dptr(), ectrl_u4->dptr(), sizeof(u4), hf_inlen);

    if (identical)
      printf(
          "[psz::info::huffman] decoded is identical to the input (quant "
          "code).\n");
    else
      printf(
          "[psz::ERR::huffman] decoded is NOT IDENTICAL to the input (quant "
          "code).\n");

    // printf("data original inlen:\t%lu\n", len);
    // printf("Huffman inlen:\t%lu\n", hf_inlen);
    // printf("Huffman outlen:\t%lu\n", hf_outlen);
    // printf(
    //     "Huffman CR = sizeof(E)*ori_len / (hf_out_bytes + outlier-overhead)
    //     = "
    //     "\t%.2lf\n",
    //     len * sizeof(E) * 1.0 / (hf_outlen + num_outlier * 2 * 4));
  }

  demo_d_predict<Predictor, T, E, H>(mem, eb, radius, stream, proto);
  psz::eval_dataquality_gpu(
      mem->xd->dptr(), mem->od->dptr(), len /*, hf_outlen */);

  GpuStreamDestroy(stream);

  delete ectrl_u4, delete ectrl_u4_decoded;
  delete mem;
}

int main(int argc, char** argv)
{
  //// help
  if (argc < 6) {
    printf(
        "0     1              2  3  4  5   [6:1   [7:lorenzo  [8:F     "
        "[9:u4  "
        "  "
        "[10:512  [11:no\n");
    printf(
        "PROG  /path/to/file  X  Y  Z  EB  [MODE  [PREDICTOR  [D-type  "
        "[E-type  "
        "[RADIUS  [USE PROTO\n");
    printf("\n");
    printf("[6] `abs` for ABS/absolute; `rel` for REL/relative to range\n");
    printf("[7] `lorenzo` or `spline3`\n");
    printf("[8] D-type (input): \"f\" for `f4`, \"d\" for `f8`\n");
    printf("[9] E-type (quant-code): \"u{1,2,4}\" for `uint{8,16,32}_t`\n");
    exit(0);
  }

  //// read argv
  auto fname = argv[1];
  auto x = atoi(argv[2]);
  auto y = atoi(argv[3]);
  auto z = atoi(argv[4]);
  auto eb = atof(argv[5]);
  eb_str = string(argv[5]);
  mode = string("rel");
  dtype = string("f");
  etype = string("u4");
  pred = string("lorenzo");
  int r = 512;
  bool useproto = false;

  if (argc > 6) mode = string(argv[6]);
  if (argc > 7) pred = string(argv[7]);
  if (argc > 8) dtype = string(argv[8]);
  if (argc > 9) etype = string(argv[9]);
  if (argc > 10) r = atoi(argv[10]);
  if (argc > 11) useproto = string(argv[11]) == "yes";

  //// dispatch
  int m = (mode == "rel") ? REL : ABS;
  int p = (pred == "spline" or pred == "spline3") ? SPLINE3 : LORENZO;
  // if (p == SPLINE3) {
  //   etype = "f4";
  //   if (dtype == "f")
  //     demo_pipeline<SPLINE3, f4, u4>(fname, x, y, z, eb, m, r);
  //   else if (dtype == "d")
  //     demo_pipeline<SPLINE3, f8, u4>(fname, x, y, z, eb, m, r);
  // }
  // else {
  if (dtype == "f") {
    demo_pipeline<LORENZO, f4, u4>(fname, x, y, z, eb, m, r, useproto);
  }
  else if (dtype == "d") {
    demo_pipeline<LORENZO, f8, u4>(fname, x, y, z, eb, m, r, useproto);
  }
  else
    throw std::runtime_error("[psz::ERR::bin_pipeline] not a valid dtype.");
  // }

  return 0;
}
