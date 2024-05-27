/**
 * @file test_l2_histsp.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-20
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "busyheader.hh"
#include "kernel/detail/histsp.dp.inl"
#include "kernel/hist.hh"
#include "kernel/histsp.hh"
#include "mem/memseg_cxx.hh"

using T = uint32_t;
using FQ = uint32_t;

constexpr auto R = 2;
constexpr auto K = 2 * R + 1;

float dist1[] = {0.01, 0.09, 0.8, 0.09, 0.01};
float dist2[] = {0.01, 0.04, 0.9, 0.04, 0.01};
float dist3[] = {0.005, 0.015, 0.96, 0.015, 0.005};

bool test1_debug()
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  auto inlen = 256;
  auto NSYM = 1024;

  auto in = new pszmem_cxx<T>(inlen, 1, 1, "hist-in");
  auto o_gpusp = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_gpusp");
  auto o_serial = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_gpusp");

  in->control({Malloc, MallocHost});
  o_gpusp->control({Malloc, MallocHost});
  o_serial->control({MallocHost});

  for (auto i = 0; i < inlen; i++) {
    in->hptr(i) = 512;
    if (i > 1 and i % 5 == 0) in->hptr(i) = 511, in->hptr(i - 1) = 513;
    if (i > 1 and i % 20 == 0) in->hptr(i) = 510, in->hptr(i - 1) = 514;
    if (i > 1 and i % 40 == 0) in->hptr(i) = 509, in->hptr(i - 1) = 515;
    if (i > 1 and i % 50 == 0) in->hptr(i) = 507, in->hptr(i - 1) = 516;
  }

  in->control({H2D});
  // float __t;

  float t_histsp_ser, t_histsp_cuda;

  dpct::queue_ptr stream;
  stream = dev_ct1.create_queue();

  psz::histsp<pszpolicy::SEQ, T, uint32_t>(
      in->hptr(), inlen, o_serial->hptr(), NSYM, &t_histsp_ser);

  psz::histsp<PROPER_GPU_BACKEND, T, uint32_t>(
      in->dptr(), inlen, o_gpusp->dptr(), NSYM, &t_histsp_cuda, stream);

  o_gpusp->control({D2H});

  // check for error
  /*
  DPCT1010:1: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 error = 0;

  auto all_eq = true;
  printf("\n\n");
  for (auto i = 0; i < NSYM; i++) {
    if (o_serial->hptr(i) != 0) {
      printf(
          "i: %d\t"
          "gpusp: %u\t"
          "serial: %u\n",
          i, o_gpusp->hptr(i), o_serial->hptr(i));
      all_eq = false;
    }
  }

  dev_ct1.destroy_queue(stream);

  delete in;
  delete o_gpusp;
  delete o_serial;

  return all_eq;
}

void helper_generate_array(
    T* in, size_t inlen, float dist[], int distlen = 5, int offset = 512)
{
  // cout << "offset: " << offset << endl;

  auto R = (distlen - 1) / 2;

  std::random_device rd;   // a seed source for the random number engine
  std::mt19937 gen(rd());  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0, inlen);

  for (auto _ = 0; _ < inlen; _++) { in[_] = offset; }
  for (auto i = 0; i < distlen; i++) {
    if (i - R == 0)
      continue;
    else {
      auto N = (int)(inlen * dist[i]);
      auto sym = (i - R) + offset;
      // printf("sym: %d, num: %d\n", sym, N);
      for (auto _ = 0; _ < N; _++) {
        auto loc = distrib(gen);
        in[loc] = sym;
      }
    }
  }
}

template <int NSYM = 1024>
bool test2_fulllen_input(size_t inlen, float gen_dist[], int distlen = K)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  auto in = new pszmem_cxx<T>(inlen, 1, 1, "hist-in");
  auto o_gpu = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_gpu");
  auto o_gpusp = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_gpusp");
  auto o_serial = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_serial");

  in->control({Malloc, MallocHost});
  o_gpu->control({Malloc, MallocHost});
  o_gpusp->control({Malloc, MallocHost});
  o_serial->control({MallocHost});

  // setup using randgen
  helper_generate_array(in->hptr(), inlen, gen_dist, distlen, NSYM / 2);

  in->control({H2D});
  float t_hist_cuda, t_histsp_ser, t_histsp_cuda;

  dpct::queue_ptr stream;
  stream = dev_ct1.create_queue();

  psz::histsp<PROPER_GPU_BACKEND, T, uint32_t>(
      in->dptr(), inlen, o_gpusp->dptr(), NSYM, &t_histsp_cuda, stream);
  // psz::histogram<PROPER_GPU_BACKEND, T>(
  //     in->dptr(), inlen, o_gpu->dptr(), NSYM, &t_hist_cuda, stream);

  psz::histsp<pszpolicy::SEQ, T, uint32_t>(
      in->hptr(), inlen, o_serial->hptr(), NSYM, &t_histsp_ser);

  o_gpu->control({D2H});
  o_gpusp->control({D2H});

  // check for error
  /*
  DPCT1010:3: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 error = 0;

  // check correctness
  auto all_eq = true;

  for (auto i = 0; i < NSYM; i++) {
    if (o_gpu->hptr(i) == o_gpusp->hptr(i) and
        o_gpusp->hptr(i) == o_serial->hptr(i)) {
      continue;
    }
    else {
      printf(
          "first not equal\t"
          "idx: %d\tgpu: %u\tgpusp: %u\tserial: %u\n",  //
          i, o_gpu->hptr(i), o_gpusp->hptr(i), o_serial->hptr(i));
      all_eq = false;
      break;
    }
  }
  if (all_eq) printf("full-length test: all equal\n");

  dev_ct1.destroy_queue(stream);

  delete in;
  delete o_gpu;
  delete o_gpusp;
  delete o_serial;

  return all_eq;
}

template <int NSYM = 1024, int CHUNK = 32768, int NWARP = 8>
bool perf(
    pszmem_cxx<T>* in, pszmem_cxx<FQ>* o_gpusp,       // for histsp
    pszmem_cxx<FQ>* o_gpu, pszmem_cxx<FQ>* o_serial,  // reference
    dpct::queue_ptr stream)
{
  constexpr auto NTREAD = 32 * NWARP;

  auto q = (sycl::queue*)stream;

  sycl::event e = q->submit([&](sycl::handler& cgh) {
    auto in_dptr_ct0 = in->dptr();
    auto in_len_ct1 = in->len();
    auto o_gpusp_dptr_ct2 = o_gpusp->dptr();

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, (in->len() - 1) / CHUNK + 1) *
                sycl::range<3>(1, 1, NTREAD),
            sycl::range<3>(1, 1, NTREAD)),
        [=](sycl::nd_item<3> item_ct1) {
          histsp_multiwarp<T, NWARP, CHUNK, FQ>(
              in_dptr_ct0, in_len_ct1, o_gpusp_dptr_ct2, NSYM, NSYM / 2);
        });
  });

  q->wait();

  // check for error
  /*
  DPCT1010:5: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 error = 0;

  // check correctness
  auto all_eq = true;

  for (auto i = 0; i < NSYM; i++) {
    if (o_gpu->hptr(i) == o_gpusp->hptr(i) and
        o_gpusp->hptr(i) == o_serial->hptr(i)) {
      continue;
    }
    else {
      printf(
          "first not equal\t"
          "idx: %d\tgpu: %u\tgpusp: %u\tserial: %u\n",  //
          i, o_gpu->hptr(i), o_gpusp->hptr(i), o_serial->hptr(i));
      all_eq = false;
      break;
    }
  }
  if (all_eq) printf("perf test: all equal\n");

  return all_eq;
}

template <int NSYM = 1024>
bool test3_performance_tuning(size_t inlen, float gen_dist[], int distlen = K)
{
  auto in = new pszmem_cxx<T>(inlen, 1, 1, "hist-in");
  auto o_gpu = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_gpu");
  auto o_gpusp = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_gpusp");
  auto o_serial = new pszmem_cxx<FQ>(NSYM, 1, 1, "hist-o_serial");

  in->control({Malloc, MallocHost});
  o_gpu->control({Malloc, MallocHost});
  o_gpusp->control({Malloc, MallocHost});
  o_serial->control({MallocHost});

  // setup using randgen
  helper_generate_array(in->hptr(), inlen, gen_dist, distlen, NSYM / 2);
  in->control({H2D});

  float t_hist_gpu, t_histsp_ser;

  // dpct::device_ext& dev_ct1 = dpct::get_current_device();
  // dpct::queue_ptr stream;
  // stream = dev_ct1.create_queue();
  sycl::queue q;

  // run CPU and GPU reference
  // psz::histogram<PROPER_GPU_BACKEND, T>(
  //     in->dptr(), inlen, o_gpu->dptr(), NSYM, &t_hist_gpu, &q);

  psz::histsp<pszpolicy::SEQ, T, uint32_t>(
      in->hptr(), inlen, o_serial->hptr(), NSYM, &t_histsp_ser);

// start testing & profiling
#define PERF(NSYM, CHUNK, NWARP) \
  eq = eq and perf<NSYM, CHUNK, NWARP>(in, o_gpusp, o_gpu, o_serial, &q);

  auto eq = true;
  PERF(NSYM, 16384, 1);
  PERF(NSYM, 16384, 2);
  PERF(NSYM, 16384, 4);
  PERF(NSYM, 16384, 8);
  PERF(NSYM, 16384, 16);
  PERF(NSYM, 16384, 32);

  PERF(NSYM, 32768, 1);
  PERF(NSYM, 32768, 2);
  PERF(NSYM, 32768, 4);
  PERF(NSYM, 32768, 8);
  PERF(NSYM, 32768, 16);
  PERF(NSYM, 32768, 32);

  PERF(NSYM, 65536, 1);
  PERF(NSYM, 65536, 2);
  PERF(NSYM, 65536, 4);
  PERF(NSYM, 65536, 8);
  PERF(NSYM, 65536, 16);
  PERF(NSYM, 65536, 32);

  PERF(NSYM, 65536 * 2, 1);
  PERF(NSYM, 65536 * 2, 2);
  PERF(NSYM, 65536 * 2, 4);
  PERF(NSYM, 65536 * 2, 8);
  PERF(NSYM, 65536 * 2, 16);
  PERF(NSYM, 65536 * 2, 32);

  // dev_ct1.destroy_queue(stream);
  delete in;
  delete o_gpu;
  delete o_gpusp;
  delete o_serial;

#undef PERF

  return eq;
}
