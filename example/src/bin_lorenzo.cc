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

#include "kernel/l23.hh"
#include "kernel/lproto.hh"
#include "stat/compare_gpu.hh"
#include "utils/io.hh"
#include "utils/print_gpu.hh"
#include "utils/viewer.hh"
#include "utils2/memseg_cxx.hh"

using std::string;

#define ABS 0
#define REL 1

string type_str;

template <typename T, typename E>
void f_lorenzo(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    double eb = 1.2e-4, int mode = REL, int const radius = 128,
    bool use_proto = false)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;

  auto radius_legal = [&](int const sizeof_T) {
    size_t upper_bound = 1lu << (sizeof_T * 8);
    cout << upper_bound << endl;
    cout << radius * 2 << endl;
    if ((radius * 2) > upper_bound)
      throw std::runtime_error("Radius overflows error-quantization type.");
  };

  radius_legal(sizeof(E));

  auto len = x * y * z;
  dim3 len3 = dim3(x, y, z);
  dim3 dummy_len3 = dim3(0, 0, 0);

  double _1, _2, rng;
  // for scanning: input->extrema_scan(_1, _2, rng); ctx->eb *= rng;

  auto oridata = new pszmem_cxx<T>(x, y, z, "oridata");
  auto de_data = new pszmem_cxx<T>(x, y, z, "de_data");
  auto errctrl = new pszmem_cxx<E>(x, y, z, "errctrl");
  auto outlier = new pszmem_cxx<T>(x, y, z, "outlier");
  oridata->control({Malloc, MallocHost})
      ->file(ifn, FromFile)
      ->control({H2D})
      ->extrema_scan(_1, _2, rng);
  de_data->control({Malloc, MallocHost});
  errctrl->control({Malloc, MallocHost});
  outlier->control({Malloc, MallocHost});

  /* a casual peek */
  printf("peeking data, 20 elements\n");
  psz::peek_device_data<T>(oridata->dptr(), 20);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float time;

  if (mode == REL) eb *= rng;

  {
    // print status
    printf("fname=\"%s\"\n", ifn);
    printf("(x, y, z)=(%lu, %lu, %lu)\n", x, y, z);
    printf(
        "(eb, mode, radius)=(%lf, %s, %d)\n", eb, mode == REL ? "REL" : "ABS",
        radius);
    printf("use proto Lorenzo=%d\n", use_proto);
    if (mode == REL) printf("eb_rel=%lf\n", eb);
  }

  if (not use_proto) {
    cout << "using optimized comp. kernel\n";
    psz_comp_l23<T, E, FP>(
        oridata->dptr(), len3, eb, radius, errctrl->dptr(), outlier->dptr(),
        &time, stream);
  }
  else {
    cout << "using proto comp. kernel\n";
    psz_comp_lproto<T, E>(
        oridata->dptr(), len3, eb, radius, errctrl->dptr(), outlier->dptr(),
        &time, stream);
  }

  cudaStreamSynchronize(stream);

  psz::peek_device_data<E>(errctrl->dptr(), 20);

  errctrl->control({D2H})->file(
      string(string(ifn) + ".eq." + type_str).c_str(), ToFile);

  if (not use_proto) {
    cout << "using optimized decomp. kernel\n";
    psz_decomp_l23<T, E, FP>(  //
        errctrl->dptr(), len3, outlier->dptr(), eb,
        radius,  // input (config)
        de_data->dptr(), &time, stream);
  }
  else {
    cout << "using prototype decomp. kernel\n";
    psz_decomp_lproto<T>(  //
        errctrl->dptr(), len3, outlier->dptr(), eb,
        radius,  // input (config)
        de_data->dptr(), &time, stream);
  }

  cudaDeviceSynchronize();

  /* demo: offline checking (de)compression quality. */
  /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(
      de_data->dptr(), oridata->dptr(), len);

  cudaStreamDestroy(stream);

  /* a casual peek */
  printf("peeking xdata, 20 elements\n");
  psz::peek_device_data<T>(de_data->dptr(), 20);

  delete oridata;
  delete de_data;
  delete errctrl;
  delete outlier;
}

int main(int argc, char** argv)
{
  //// help
  if (argc < 6) {
    printf(
        "0     1              2  3  4  5   [6:1]   [7:F]     [8:u4]    "
        "[9:512]   [10:no]\n");
    printf(
        "PROG  /path/to/file  X  Y  Z  EB  [MODE]  [D-type]  [E-type]  "
        "[Radius]  [Use Prototype]\n");
    printf("\n");
    printf("[6] `abs` for ABS/absolute; `rel` for REL/relative to range\n");
    printf("[7] D-type: \"f\" for `float`, \"d\" for `double`\n");
    printf(
        "[8] E-type: \"u{1,2,4}\" for `uint{8,16,32}_t` as quant-code "
        "type\n");
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
  int radius = 512;
  bool use_proto = false;

  if (argc > 6) mode_str = string(argv[6]);
  if (argc > 7) etype = string(argv[7]);
  if (argc > 8) radius = atoi(argv[8]);
  if (argc > 9) use_proto = string(argv[9]) == "yes";

  //// dispatch
  type_str = etype;
  int mode = (mode_str == "rel") ? REL : ABS;

  if (dtype == "f") {
    if (etype == "u1")
      f_lorenzo<float, uint8_t>(fname, x, y, z, eb, mode, radius, use_proto);
    else if (etype == "u2")
      f_lorenzo<float, uint16_t>(fname, x, y, z, eb, mode, radius, use_proto);
    else if (etype == "u4")
      f_lorenzo<float, uint32_t>(fname, x, y, z, eb, mode, radius, use_proto);
  }
  else if (dtype == "d") {
    if (etype == "u1")
      f_lorenzo<double, uint8_t>(fname, x, y, z, eb, mode, radius, use_proto);
    else if (etype == "u2")
      f_lorenzo<double, uint16_t>(fname, x, y, z, eb, mode, radius, use_proto);
    else if (etype == "u4")
      f_lorenzo<double, uint32_t>(fname, x, y, z, eb, mode, radius, use_proto);
  }
  else
    throw std::runtime_error("not a valid dtype.");

  return 0;
}
