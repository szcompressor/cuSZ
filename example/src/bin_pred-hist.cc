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

#include "kernel/l23.hh"
#include "kernel/lproto.hh"
#include "kernel/spline.hh"
#include "kernel2/histsp.hh"
#include "stat/compare_gpu.hh"
#include "stat/stat.hh"
#include "utils/print_gpu.hh"
#include "utils/timer.hh"
#include "utils/viewer.hh"
#include "utils2/memseg_cxx.hh"

using std::string;
using std::to_string;

#define ABS 0
#define REL 1

#define LORENZO 11
#define SPLINE3 12

string type_str;

template <int predictor = LORENZO, typename T = float, typename E = uint32_t>
void predict_demo(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    double eb = 1.2e-4, int mode = REL, int const radius = 128,
    bool use_proto = false)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;

  auto const ori_eb = eb;

  auto radius_legal = [&](int const sizeof_T) {
    size_t upper_bound = 1lu << (sizeof_T * 8);
    // cout << upper_bound << endl;
    // cout << radius * 2 << endl;
    if ((radius * 2) > upper_bound)
      throw std::runtime_error("Radius overflows error-quantization type.");
  };
  auto ndim = [](dim3 l) {
    auto n = 3;
    if (l.z == 1) n = 2;
    if (l.y == 1) n = 1;
    return n;
  };

  radius_legal(sizeof(E));

  auto len = x * y * z;
  dim3 len3 = dim3(x, y, z);
  dim3 dummy_len3 = dim3(0, 0, 0);

  double _1, _2, rng;

  auto oridata = new pszmem_cxx<T>(x, y, z, "oridata");
  auto de_data = new pszmem_cxx<T>(x, y, z, "de_data");

  auto e_space = new pszmem_cxx<E>(len * 1.5, 1, 1, "raw space for ectrl");

  auto s3l3 = new Spline3Len3;
  spline3_calc_sizes((void*)&len3, s3l3);

  auto ectrl_l = new pszmem_cxx<E>(x, y, z, "ectrl_l");

  auto ectrl_s =
      new pszmem_cxx<E>(s3l3->x_32, s3l3->y_8, s3l3->z_8, "ectrl_s");
  auto anchor = new pszmem_cxx<T>(
      s3l3->l3_anchor.x, s3l3->l3_anchor.y, s3l3->l3_anchor.z, "anchor");

  auto outlier = new pszmem_cxx<T>(x, y, z, "outlier");

  auto hist_q1 = new pszmem_cxx<uint32_t>(radius * 2, 1, 1, "hist-normal");
  auto hist_q2 = new pszmem_cxx<uint32_t>(radius * 2, 1, 1, "hist-sp");

  oridata->control({Malloc, MallocHost})
      ->file(ifn, FromFile)
      ->control({H2D})
      ->extrema_scan(_1, _2, rng);
  de_data->control({Malloc, MallocHost});

  e_space->control({Malloc, MallocHost});
  ectrl_l->asaviewof(e_space);
  ectrl_s->asaviewof(e_space);

  anchor->control({Malloc, MallocHost});

  outlier->control({Malloc, MallocHost});
  hist_q1->control({MallocHost});
  hist_q2->control({MallocHost});

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float time, time_hist1, time_hist2;

  if (mode == REL) eb *= rng;

  {
    // print status
    printf("\033[1mfname=\033[0m\"%s\"\n", ifn);
    printf("\033[1m(x, y, z)=\033[0m(%lu, %lu, %lu)\n", x, y, z);
    printf(
        "\033[1m(eb, mode, radius)=\033[0m(%lf, %s, %d)\n", eb,
        mode == REL ? "REL" : "ABS", radius);
    printf(
        "\033[1mpredictor=\033[0m%s\n",
        predictor == SPLINE3 ? "spline3" : "lorenzo");
    printf("\033[1muse proto Lorenzo=\033[0m%s\n", use_proto ? "yes" : "no");
    if (mode == REL) printf("\033[1meb_rel=\033[0m%lf\n", eb);
  }

  if (predictor == LORENZO) {
    if (not use_proto) {
      psz_comp_l23<T, E, FP>(
          oridata->dptr(), len3, eb, radius, ectrl_l->dptr(), outlier->dptr(),
          &time, stream);
    }
    else {
      psz_comp_lproto<T, E>(
          oridata->dptr(), len3, eb, radius, ectrl_l->dptr(), outlier->dptr(),
          &time, stream);
    }
  }
  else if (predictor == SPLINE3) {
    if (ndim(len3) != 3) throw std::runtime_error("SPLINE3: must be 3D data.");
    spline_construct(oridata, anchor, ectrl_s, eb, radius, stream);
  }
  else {
    throw std::runtime_error("Must be LORENZO or SPLINE3.");
  }

  cudaStreamSynchronize(stream);

  ectrl_l->control({D2H})->file(
      string(string(ifn) + ".eq." + type_str + ".eb." + to_string(ori_eb))
          .c_str(),
      ToFile);

  psz::stat::histogram<psz_policy::CPU, E>(
      ectrl_l->hptr(), len, hist_q1->hptr(), radius * 2, &time_hist1);

  cout << "time_hist normal:\t" << time_hist1 << endl;

  auto a = hires::now();
  histsp<psz_policy::CPU, E, uint32_t>(
      ectrl_l->hptr(), len, hist_q2->hptr(), radius * 2);
  auto b = hires::now();
  cout << "time_hist sp:\t" << static_cast<duration_t>(b - a).count() * 1000
       << endl;

  hist_q1->file(
      string(string(ifn) + ".hist." + type_str + ".eb." + to_string(ori_eb))
          .c_str(),
      ToFile);

  if (predictor == LORENZO) {
    if (not use_proto) {
      psz_decomp_l23<T, E, FP>(  //
          ectrl_l->dptr(), len3, outlier->dptr(), eb,
          radius,  // input (config)
          de_data->dptr(), &time, stream);
    }
    else {
      psz_decomp_lproto<T>(  //
          ectrl_l->dptr(), len3, outlier->dptr(), eb,
          radius,  // input (config)
          de_data->dptr(), &time, stream);
    }
  }
  else if (predictor == SPLINE3) {
    if (ndim(len3) != 3) throw std::runtime_error("SPLINE3: must be 3D data.");
    spline_reconstruct(anchor, ectrl_s, de_data, eb, radius, stream);
  }
  else {
    throw std::runtime_error("Must be LORENZO or SPLINE3.");
  }

  cudaDeviceSynchronize();

  /* demo: offline checking (de)compression quality. */
  /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(
      de_data->dptr(), oridata->dptr(), len);

  cudaStreamDestroy(stream);

  delete oridata;
  delete de_data;
  delete e_space;
  delete anchor;
  delete outlier;
}

int main(int argc, char** argv)
{
  //// help
  if (argc < 6) {
    printf(
        "0     1              2  3  4  5   [6:1   [7:lorenzo  [8:F     [9:u4  "
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
  auto mode_str = string("rel");
  auto dtype = string("f");
  auto etype = string("u4");
  auto predictor_str = string("lorenzo");
  int r = 512;
  bool useproto = false;

  if (argc > 6) mode_str = string(argv[6]);
  if (argc > 7) predictor_str = string(argv[7]);
  if (argc > 8) etype = string(argv[8]);
  if (argc > 9) etype = string(argv[9]);
  if (argc > 10) r = atoi(argv[10]);
  if (argc > 11) useproto = string(argv[11]) == "yes";

  //// dispatch
  type_str = etype;
  int m = (mode_str == "rel") ? REL : ABS;
  int p = (predictor_str == "spline" or predictor_str == "spline3") ? SPLINE3
                                                                    : LORENZO;
  if (p == SPLINE3) {
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
      else if (etype == "u4")
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
      else if (etype == "u4")
        predict_demo<LORENZO, double, uint32_t>(
            fname, x, y, z, eb, m, r, useproto);
    }
    else
      throw std::runtime_error("not a valid dtype.");
  }

  return 0;
}
