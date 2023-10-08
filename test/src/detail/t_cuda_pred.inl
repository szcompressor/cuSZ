/**
 * @file test_l3_cuda_pred.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <typeinfo>

#include "kernel/lrz.hh"
#include "mem/memseg_cxx.hh"
#include "stat/compare/compare.stl.hh"
#include "stat/compare/compare.thrust.hh"
#include "utils/print_arr.hh"
#include "utils/viewer.hh"

std::string type_literal;

template <typename T, typename Eq, bool LENIENT = true>
bool f(
    size_t const x, size_t const y, size_t const z, double const eb,
    int const radius = 512)
{
  // When the input type is FP<X>, the internal precision should be the same.
  using FP = T;
  auto len = x * y * z;

  auto oridata = new pszmem_cxx<T>(x, y, z, "oridata");
  auto de_data = new pszmem_cxx<T>(x, y, z, "de_data");
  auto outlier = new pszmem_cxx<T>(x, y, z, "outlier");
  auto errctrl = new pszmem_cxx<Eq>(x, y, z, "errctrl");

  oridata->control({MallocManaged});
  de_data->control({MallocManaged});
  outlier->control({MallocManaged});
  errctrl->control({MallocManaged});

  psz::testutils::cu_hip::rand_array<T>(oridata->uniptr(), len);

  GpuStreamT stream;
  GpuStreamCreate(&stream);

  float time;
  auto len3 = dim3(x, y, z);

  psz_comp_l23<T, Eq, FP>(                   //
      oridata->uniptr(), len3, eb, radius,   // input and config
      errctrl->uniptr(), outlier->uniptr(),  // output
      &time, stream);
  GpuStreamSync(stream);

  psz_decomp_l23<T, Eq, FP>(                       //
      errctrl->uniptr(), len3, outlier->uniptr(),  // input
      eb, radius,                                  // input (config)
      de_data->uniptr(),                           // output
      &time, stream);
  GpuStreamSync(stream);

  // psz::peek_data(oridata->uniptr(), 100);
  // psz::peek_data(de_data->uniptr(), 100);

  size_t first_non_eb = 0;
  // bool   error_bounded = psz::thrustgpu::thrustgpu_error_bounded<T>(de_data,
  // oridata, len, eb, &first_non_eb);
  bool error_bounded = psz::cppstl_error_bounded<T>(
      de_data->uniptr(), oridata->uniptr(), len, eb, &first_non_eb);

  // psz::eval_dataquality_gpu(oridata->uniptr(), de_data->uniptr(), len);

  GpuStreamDestroy(stream);
  delete oridata;
  delete de_data;
  delete errctrl;
  delete outlier;

  printf(
      "(%zu,%zu,%zu)\t(T=%s,Eq=%s)\terror bounded?\t", x, y, z,
      typeid(T).name(), typeid(Eq).name());
  if (not LENIENT) {
    if (not error_bounded) throw std::runtime_error("NO");
  }
  else {
    cout << (error_bounded ? "yes" : "NO") << endl;
  }

  return error_bounded;
}

bool g(uint32_t x, uint32_t y, uint32_t z)
{
  auto all_pass = true;
  double eb = 1e-4;

  all_pass = all_pass and f<float, uint8_t>(x, y, z, eb, 128);
  all_pass = all_pass and f<float, uint16_t>(x, y, z, eb, 512);
  all_pass = all_pass and f<float, uint32_t>(x, y, z, eb, 512);
  all_pass = all_pass and f<float, float>(x, y, z, eb, 512);
  all_pass = all_pass and f<double, uint8_t>(x, y, z, eb, 128);
  all_pass = all_pass and f<double, uint16_t>(x, y, z, eb, 512);
  all_pass = all_pass and f<double, uint32_t>(x, y, z, eb, 512);
  all_pass = all_pass and f<double, float>(x, y, z, eb, 512);

  all_pass = all_pass and f<float, int32_t>(x, y, z, eb, 512);
  all_pass = all_pass and f<double, int32_t>(x, y, z, eb, 512);

  return all_pass;
}
