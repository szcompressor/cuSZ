/**
 * @file demo_lorenzo.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <stdexcept>
#include <string>
#include <type_traits>

#include "kernel/l23.hh"
#include "kernel/lproto.hh"
#include "kernel/spline.hh"
#include "kernel/histsp.hh"
#include "mem/layout_cxx.hh"
#include "mem/memseg_cxx.hh"
#include "stat/compare_gpu.hh"
#include "stat/stat.hh"
#include "utils/print_gpu.hh"
#include "utils/timer.hh"
#include "utils/viewer.hh"

using std::string;
using std::to_string;

#define ABS 0
#define REL 1

#define LORENZO 11
#define SPLINE3 12

#define PRINT_STATUS                                                          \
  {                                                                           \
    printf("\033[1mfname=\033[0m\"%s\"\n", ifn);                              \
    printf("\033[1m(x, y, z)=\033[0m(%lu, %lu, %lu)\n", x, y, z);             \
    printf(                                                                   \
        "\033[1m(eb, mode, radius)=\033[0m(%lf, %s, %d)\n", eb,               \
        mode == REL ? "REL" : "ABS", radius);                                 \
    printf(                                                                   \
        "\033[1mpredictor=\033[0m%s\n",                                       \
        predictor == SPLINE3 ? "spline3" : "lorenzo");                        \
    printf("\033[1muse proto Lorenzo=\033[0m%s\n", use_proto ? "yes" : "no"); \
    if (mode == REL) printf("\033[1meb_rel=\033[0m%lf\n", eb);                \
  }

string dtype, etype, mode, pred, eb_str;
size_t len;
dim3 len3, dummy_len3;
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
    int predictor = LORENZO, typename T = float, typename E = uint32_t,
    typename H = uint32_t>
void predict_demo(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    double eb = 1.2e-4, int mode = REL, int const radius = 128,
    bool use_proto = false)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;
  using M = uint32_t;

  radius_legal(radius, sizeof(E));

  len = x * y * z;
  len3 = dim3(x, y, z);

  auto mem = new pszmempool_cxx<T, E, H>(x, radius, y, z);
  mem->od->control({Malloc, MallocHost})
      ->file(ifn, FromFile)
      ->control({H2D})
      ->extrema_scan(_1, _2, rng);
  mem->xd->control({Malloc, MallocHost});

  auto len3p = mem->es->template len3<dim3>();
  auto ectrl_su4 =
      predictor == SPLINE3
          ? new pszmem_cxx<_u4>(len3p.x, len3p.y, len3p.z, "ectrl_su4")
          : new pszmem_cxx<_u4>(x, y, z, "ectrl_su4");

  auto hist_q1 = new pszmem_cxx<_u4>(radius * 2, 1, 1, "hist-normal");
  auto hist_q2 = new pszmem_cxx<_u4>(radius * 2, 1, 1, "hist-sp");

  ectrl_su4->control({Malloc, MallocHost});
  hist_q1->control({MallocHost});
  hist_q2->control({MallocHost});

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float time, time_hist1, time_hist2;

  if (mode == REL) eb *= rng;

  PRINT_STATUS;

  if (predictor == LORENZO) {
    if (not use_proto) {
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
    mem->el->castto(ectrl_su4, psz_space::Host)->control({H2D});
  }
  else if (predictor == SPLINE3) {
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
    mem->es->castto(ectrl_su4, psz_space::Host)->control({H2D});

    ectrl_su4->file(
        string(string(ifn) + ".eqcompat." + suffix(true)).c_str(), ToFile);
  }
  else {
    throw std::runtime_error("Must be LORENZO or SPLINE3.");
  }

  {  // histogram

    auto hist_in = predictor == SPLINE3 ? mem->es : mem->el;

    psz::stat::histogram<psz_policy::CPU, E>(
        hist_in->hptr(), hist_in->len(), hist_q1->hptr(), radius * 2,
        &time_hist1);

    cout << "time_hist normal:\t" << time_hist1 << endl;

    auto a = hires::now();
    histsp<psz_policy::CPU, E, uint32_t>(
        hist_in->hptr(), hist_in->len(), hist_q2->hptr(), radius * 2);
    auto b = hires::now();
    cout << "time_hist sp:\t" << static_cast<duration_t>(b - a).count() * 1000
         << endl;

    hist_q1->file(string(string(ifn) + ".ht." + suffix()).c_str(), ToFile);
  }

  if (predictor == LORENZO) {
    if (not use_proto) {
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
  else if (predictor == SPLINE3) {
    if (ndim(len3) != 3) throw std::runtime_error("SPLINE3: must be 3D data.");
    spline_reconstruct(mem->ac, mem->es, mem->xd, eb, radius, stream);
  }
  else {
    throw std::runtime_error("Must be LORENZO or SPLINE3.");
  }

  cudaDeviceSynchronize();

  /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(
      mem->xd->dptr(), mem->od->dptr(), len);

  cudaStreamDestroy(stream);

  delete ectrl_su4, delete hist_q1, delete hist_q2;
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
      predict_demo<SPLINE3, float, float>(fname, x, y, z, eb, m, r);
    else if (dtype == "d")
      predict_demo<SPLINE3, double, float>(fname, x, y, z, eb, m, r);
  }
  else {
    if (dtype == "f") {
      if (etype == "u1")
        predict_demo<LORENZO, float, uint8_t>(
            fname, x, y, z, eb, m, r, useproto);
      else if (etype == "u2")
        predict_demo<LORENZO, float, uint16_t>(
            fname, x, y, z, eb, m, r, useproto);
      if (etype == "u4")
        predict_demo<LORENZO, float, uint32_t>(
            fname, x, y, z, eb, m, r, useproto);
    }
    else if (dtype == "d") {
      if (etype == "u1")
        predict_demo<LORENZO, double, uint8_t>(
            fname, x, y, z, eb, m, r, useproto);
      else if (etype == "u2")
        predict_demo<LORENZO, double, uint16_t>(
            fname, x, y, z, eb, m, r, useproto);
      if (etype == "u4")
        predict_demo<LORENZO, double, uint32_t>(
            fname, x, y, z, eb, m, r, useproto);
    }
    else
      throw std::runtime_error("not a valid dtype.");
  }

  return 0;
}
