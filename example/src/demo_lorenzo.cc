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

std::string type_literal;

template <typename T, typename E>
void f(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    double const eb = 1.2e-4, int const radius = 128, bool use_proto = false)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;

  auto len = x * y * z;
  dim3 len3 = dim3(x, y, z);
  dim3 dummy_len3 = dim3(0, 0, 0);

  auto oridata = new pszmem_cxx<T>(x, y, z, "oridata");
  auto de_data = new pszmem_cxx<T>(x, y, z, "de_data");
  auto errctrl = new pszmem_cxx<E>(x, y, z, "errctrl");
  auto outlier = new pszmem_cxx<T>(x, y, z, "outlier");
  oridata->control({Malloc, MallocHost})->file(ifn, FromFile)->control({H2D});
  de_data->control({Malloc, MallocHost});
  errctrl->control({Malloc, MallocHost});
  outlier->control({Malloc, MallocHost});

  /* a casual peek */
  printf("peeking data, 20 elements\n");
  psz::peek_device_data<T>(oridata->dptr(), 20);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float time;

  if (not use_proto) {
    cout << "using optimized comp. kernel\n";
    psz_comp_l23<T, E, FP>(                 //
        oridata->dptr(), len3, eb, radius,  //
        errctrl->dptr(), outlier->dptr(), &time, stream);
  }
  else {
    psz_comp_lproto<T, E>(                  //
        oridata->dptr(), len3, eb, radius,  // input and config
        errctrl->dptr(), outlier->dptr(), &time, stream);
  }

  cudaStreamSynchronize(stream);

  psz::peek_device_data<E>(errctrl->dptr(), 20);

  errctrl->control({D2H})->file(
      string(string(ifn) + ".eq." + type_literal).c_str(), ToFile);

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
    printf("0    1             2     3 4 5 6  [7]      [8:128]  [9:yes]\n");
    printf(
        "PROG /path/to/file DType X Y Z EB [EType]  [Radius] [Use "
        "Prototype]\n");
    printf(" 2  DType: \"F\" for `float`, \"D\" for `double`\n");
    printf(
        "[7] EType: \"ui{8,16,32}\" for `uint{8,16,32}_t` as quant-code "
        "type\n");
    exit(0);
  }

  //// read argv
  auto fname = argv[1];
  auto dtype = std::string(argv[2]);
  auto x = atoi(argv[3]);
  auto y = atoi(argv[4]);
  auto z = atoi(argv[5]);
  auto eb = atof(argv[6]);

  std::string etype;
  if (argc > 7)
    etype = std::string(argv[7]);
  else
    etype = "ui16";
  type_literal = etype;

  int radius;
  if (argc > 8)
    radius = atoi(argv[8]);
  else
    radius = 128;

  bool use_prototype;
  if (argc > 9)
    use_prototype = std::string(argv[9]) == "yes";
  else
    use_prototype = false;

  //// dispatch

  auto radius_legal = [&](int const sizeof_T) {
    size_t upper_bound = 1lu << (sizeof_T * 8);
    cout << upper_bound << endl;
    cout << radius * 2 << endl;
    if ((radius * 2) > upper_bound)
      throw std::runtime_error("Radius overflows error-quantization type.");
  };

  if (dtype == "F") {
    if (etype == "ui8") {
      radius_legal(1);
      f<float, uint8_t>(fname, x, y, z, eb, radius, use_prototype);
    }
    else if (etype == "ui16") {
      radius_legal(2);
      f<float, uint16_t>(fname, x, y, z, eb, radius, use_prototype);
    }
    else if (etype == "ui32") {
      radius_legal(4);
      f<float, uint32_t>(fname, x, y, z, eb, radius, use_prototype);
    }
  }
  else if (dtype == "D") {
    if (etype == "ui8") {
      radius_legal(1);
      f<double, uint8_t>(fname, x, y, z, eb, radius, use_prototype);
    }
    else if (etype == "ui16") {
      radius_legal(2);
      f<double, uint16_t>(fname, x, y, z, eb, radius, use_prototype);
    }
    else if (etype == "ui32") {
      radius_legal(4);
      f<double, uint32_t>(fname, x, y, z, eb, radius, use_prototype);
    }
  }
  else
    throw std::runtime_error("not a valid dtype.");

  return 0;
}
