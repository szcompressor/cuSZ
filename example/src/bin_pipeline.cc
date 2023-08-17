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
#include "kernel/hist.hh"
#include "kernel/histsp.hh"
#include "kernel/l23.hh"
#include "kernel/lproto.hh"
#include "kernel/spline.hh"
#include "mem/layout_cxx.hh"
#include "mem/memseg_cxx.hh"
#include "stat/compare_gpu.hh"
#include "utils/print_gpu.hh"
#include "utils/timer.hh"
#include "utils/viewer.hh"

using std::string;
using std::to_string;

#define ABS 0
#define REL 1

#define LORENZO 11
#define SPLINE3 12

#define PRINT_STATUS                                                      \
  {                                                                       \
    printf("\033[1mfname=\033[0m\"%s\"\n", ifn);                          \
    printf("\033[1m(x, y, z)=\033[0m(%lu, %lu, %lu)\n", x, y, z);         \
    printf(                                                               \
        "\033[1m(eb, mode, radius)=\033[0m(%lf, %s, %d)\n", eb,           \
        mode == REL ? "REL" : "ABS", radius);                             \
    printf(                                                               \
        "\033[1mpredictor=\033[0m%s\n",                                   \
        Predictor == SPLINE3 ? "spline3" : "lorenzo");                    \
    printf("\033[1muse proto Lorenzo=\033[0m%s\n", proto ? "yes" : "no"); \
    if (mode == REL) printf("\033[1meb_rel=\033[0m%lf\n", eb);            \
  }

string dtype, etype, mode, pred, eb_str;
double _1, _2, rng;

namespace {

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

template <
    int Predictor, typename T = float, typename E = uint32_t,
    typename H = uint32_t>
void demo_c_predict(
    pszmempool_cxx<T, E, H>* mem, pszmem_cxx<u4>* ectrl_u4, char const* ifn,
    double const eb, int const radius, cudaStream_t stream, bool proto = false)
{
  using FP = T;

  auto len3 = mem->el->template len3<dim3>();
  auto len3p = mem->es->template len3<dim3>();
  float time, time_histcpu_base, time_histcpu_optim;

  if (Predictor == LORENZO) {
    if (not proto) {
      psz_comp_l23<T, E, FP>(
          mem->od->dptr(), len3, eb, radius, mem->ectrl_lrz(),
          mem->outlier_space(), &time, stream);
    }
    else {
      psz_comp_lproto<T, E>(
          mem->od->dptr(), len3, eb, radius, mem->ectrl_lrz(),
          mem->outlier_space(), &time, stream);
    }

    mem->el
        ->control({D2H})  //
        ->file(string(string(ifn) + ".eq." + suffix()).c_str(), ToFile);
    mem->el->castto(ectrl_u4, psz_space::Host)->control({H2D});
  }
  else if (Predictor == SPLINE3) {
    if (ndim(len3) != 3) throw std::runtime_error("SPLINE3: must be 3D data.");
    spline_construct(mem->od, mem->ac, mem->es, eb, radius, stream);

    printf(
        "using spline3:\n"
        "original size (xyz order):\t%u %u %u\n"
        "padded size (xyz order):\t%u %u %u\n",
        len3.x, len3.y, len3.z, len3p.x, len3p.y, len3p.z);

    mem->es
        ->control({D2H})  //
        ->file(string(string(ifn) + ".eq." + suffix()).c_str(), ToFile);
    mem->es->castto(ectrl_u4, psz_space::Host)->control({H2D});

    ectrl_u4->file(
        string(string(ifn) + ".eqcompat." + suffix(true)).c_str(), ToFile);
  }
  else {
    throw std::runtime_error("Must be LORENZO or SPLINE3.");
  }
}

template <
    int Predictor, typename T = float, typename E = uint32_t,
    typename H = uint32_t>
void demo_d_predict(
    pszmempool_cxx<T, E, H>* mem, double const eb, int const radius,
    cudaStream_t stream, bool proto = false)
{
  using FP = T;

  auto len3 = mem->el->template len3<dim3>();
  auto len3p = mem->es->template len3<dim3>();
  float time;

  if (Predictor == LORENZO) {
    if (not proto) {
      psz_decomp_l23<T, E, FP>(
          mem->ectrl_lrz(), len3, mem->outlier_space(), eb, radius,
          mem->xd->dptr(), &time, stream);
    }
    else {
      psz_decomp_lproto<T>(
          mem->ectrl_lrz(), len3, mem->outlier_space(), eb, radius,
          mem->xd->dptr(), &time, stream);
    }
  }
  else if (Predictor == SPLINE3) {
    if (ndim(len3) != 3) throw std::runtime_error("SPLINE3: must be 3D data.");
    spline_reconstruct(mem->ac, mem->es, mem->xd, eb, radius, stream);
  }
  else {
    throw std::runtime_error("Must be LORENZO or SPLINE3.");
  }
}

template <typename H>
void demo_hist_u4in(
    pszmem_cxx<u4>* hist_in, pszmem_cxx<H>* hist_out, char const* ifn,
    int const radius, cudaStream_t stream)
{
  using E = u4;

  float tgpu_base, tgpu_optim;
  float tcpu_base, tcpu_optim;

  auto ser_base = new pszmem_cxx<u4>(radius * 2, 1, 1, "hist-normal");
  auto ser_optim = new pszmem_cxx<E>(radius * 2, 1, 1, "hist-sp_cpu");
  auto gpu_optim = new pszmem_cxx<E>(radius * 2, 1, 1, "hist-sp_gpu");
  ser_base->control({MallocHost});
  ser_optim->control({MallocHost});
  gpu_optim->control({MallocHost, Malloc});

  printf("histogram inlen:\t%lu\n", hist_in->len());

  auto bklen = radius * 2;
  hist<CPU, E>(
      false, hist_in->hptr(), hist_in->len(), ser_base->hptr(), bklen,
      &tcpu_base, stream);
  hist<CPU, E>(
      true, hist_in->hptr(), hist_in->len(), ser_optim->hptr(), bklen,
      &tcpu_optim, stream);
  hist<CUDA, E>(
      false, hist_in->dptr(), hist_in->len(), hist_out->dptr(), bklen,
      &tgpu_base, stream);
  hist<CUDA, E>(
      true, hist_in->dptr(), hist_in->len(), gpu_optim->dptr(), bklen,
      &tgpu_optim, stream);

  ser_base->file(string(string(ifn) + ".ht." + suffix()).c_str(), ToFile);

  printf(
      "hist cpu baseline:\t%5.2f ms\toptim:\t%5.2f ms (speedup: %3.2fx)\n",  //
      tcpu_base, tcpu_optim, tcpu_base / tcpu_optim);
  printf(
      "hist gpu baseline:\t%5.2f ms\toptim:\t%5.2f ms (speedup: %3.2fx)\n",  //
      tgpu_base, tgpu_optim, tgpu_base / tgpu_optim);

  delete gpu_optim;
  delete ser_base;
  delete ser_optim;
}

template <
    int Predictor = LORENZO, typename T = float, typename E = uint32_t,
    typename H = uint32_t>
void demo_pipeline(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    double eb = 1.2e-4, int mode = REL, int const radius = 128,
    bool proto = false)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;
  using M = uint32_t;

  radius_legal(radius, sizeof(E));

  auto len = x * y * z;
  auto len3 = dim3(x, y, z);
  float time, time_histcpu_base, time_histcpu_optim;
  constexpr auto pardeg = 768;

  cusz::HuffmanCodec<u4, H, u4> codec;

  auto mem = new pszmempool_cxx<T, E, H>(x, radius, y, z);
  mem->od->control({Malloc, MallocHost})
      ->file(ifn, FromFile)
      ->control({H2D})
      ->extrema_scan(_1, _2, rng);
  mem->xd->control({Malloc, MallocHost});

  auto len3p = mem->es->template len3<dim3>();
  auto ectrl_u4 =
      Predictor == SPLINE3
          ? new pszmem_cxx<u4>(len3p.x, len3p.y, len3p.z, "ectrl_u4")
          : new pszmem_cxx<u4>(x, y, z, "ectrl_u4");
  // TODO duplicate pszmem
  auto ectrl_u4_decoded =
      Predictor == SPLINE3
          ? new pszmem_cxx<u4>(len3p.x, len3p.y, len3p.z, "ectrl_u4_dec")
          : new pszmem_cxx<u4>(x, y, z, "ectrl_u4_dec");

  ectrl_u4->control({Malloc, MallocHost});
  ectrl_u4_decoded->control({Malloc, MallocHost});

  codec.init(ectrl_u4->len(), radius * 2, pardeg /* not optimal for perf */);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

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
  printf("#outlier:\t%u\n", num_outlier);

  demo_hist_u4in<H>(ectrl_u4, mem->ht, ifn, radius, stream);

  size_t hf_outlen;
  {  // Huffman
    auto hf_inlen = ectrl_u4->len();
    uint8_t* d_encoded;

    codec.build_codebook(mem->hist(), radius * 2, stream);
    codec.encode(
        ectrl_u4->dptr(), ectrl_u4->len(), &d_encoded, &hf_outlen, stream);
    codec.decode(d_encoded, ectrl_u4_decoded->dptr());

    auto identical = psz::thrustgpu_identical(
        ectrl_u4_decoded->dptr(), ectrl_u4->dptr(), hf_inlen);

    if (identical)
      printf("Huffman: decoded is identical to the input (quant code).\n");
    else
      printf(
          "Huffman raises an ERROR: decoded is NOT IDENTICAL to the input "
          "(quant code).\n");

    printf("data original inlen:\t%lu\n", len);
    printf("Huffman inlen:\t%lu\n", hf_inlen);
    printf("Huffman outlen:\t%lu\n", hf_outlen);
    printf(
        "Huffman CR = sizeof(E)*ori_len / (hf_out_bytes + outlier-overhead) = "
        "\t%.2lf\n",
        len * sizeof(E) * 1.0 / (hf_outlen + num_outlier * 2 * 4));
  }

  demo_d_predict<Predictor, T, E, H>(mem, eb, radius, stream, proto);
  psz::eval_dataquality_gpu(mem->xd->dptr(), mem->od->dptr(), len, hf_outlen);

  cudaStreamDestroy(stream);

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
    printf("[8] D-type (input): \"f\" for `float`, \"d\" for `double`\n");
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
  if (p == SPLINE3) {
    etype = "f4";
    if (dtype == "f")
      demo_pipeline<SPLINE3, float, float>(fname, x, y, z, eb, m, r);
    else if (dtype == "d")
      demo_pipeline<SPLINE3, double, float>(fname, x, y, z, eb, m, r);
  }
  else {
    if (dtype == "f") {
      if (etype == "u1")
        demo_pipeline<LORENZO, float, uint8_t>(
            fname, x, y, z, eb, m, r, useproto);
      else if (etype == "u2")
        demo_pipeline<LORENZO, float, uint16_t>(
            fname, x, y, z, eb, m, r, useproto);
      if (etype == "u4")
        demo_pipeline<LORENZO, float, uint32_t>(
            fname, x, y, z, eb, m, r, useproto);
    }
    else if (dtype == "d") {
      if (etype == "u1")
        demo_pipeline<LORENZO, double, uint8_t>(
            fname, x, y, z, eb, m, r, useproto);
      else if (etype == "u2")
        demo_pipeline<LORENZO, double, uint16_t>(
            fname, x, y, z, eb, m, r, useproto);
      if (etype == "u4")
        demo_pipeline<LORENZO, double, uint32_t>(
            fname, x, y, z, eb, m, r, useproto);
    }
    else
      throw std::runtime_error("not a valid dtype.");
  }

  return 0;
}
