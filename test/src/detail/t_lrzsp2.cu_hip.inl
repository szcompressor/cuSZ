/**
 * @file test_l3_lorenzosp.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <typeinfo>

#include "../rand.hh"
#include "cusz/nd.h"
#include "cusz/type.h"
#include "kernel/lrz.hh"
#include "kernel/spv.hh"
#include "mem.hh"
#include "stat/compare/compare.stl.hh"
#include "stat/compare/compare.thrust.hh"
#include "utils/print_arr.hh"
#include "utils/viewer.hh"

std::string type_literal;

#define MEMPOOL_GPU pszmempool_cxx<T, E, u4, PROPER_GPU_BACKEND>
#define MEMPOOL_SEQ pszmempool_cxx<T, E, u4, SEQ>
#define MEM_T pszmem_cxx<T>

template <typename T = f4>
void create_dummy_input(MEM_T* in, szt len, f4 ratio = 0.001)
{
  // setup dummy outliers
  auto ratio_outlier = 0.001;
  auto n = (int)(ratio_outlier * len);
  auto step = (int)(len / (n + 1));
  psz::testutils::cu_hip::rand_array<T>(in->dptr(), len);

  in->control({D2H});
  for (auto i = 0; i < n; i++) { in->hptr(i * step) *= 4; }
  in->control({H2D});
}

template <typename T = f4, typename E = u4, bool LENIENT = true>
bool run_seq(MEMPOOL_SEQ* mem, double const eb, int const radius)
{
  float time;

  psz_comp_l23_seq<T, E>(
      mem->od->hptr(), mem->od->template len3<psz_dim3>(), eb, radius,
      mem->e->hptr(), mem->compact, &time);

  psz::spv_scatter_naive<SEQ>(
      mem->compact_val(), mem->compact_idx(), mem->compact->num_outliers(),
      mem->xd->hptr(), &time, nullptr);
  psz::spv_scatter_naive<SEQ>(
      mem->compact_val(), mem->compact_idx(), mem->compact->num_outliers(),
      mem->xdtest->hptr(), &time, nullptr);

  psz_decomp_l23_seq<T, E>(
      mem->e->hptr(), mem->e->template len3<psz_dim3>(), mem->xd->hptr(), eb,
      radius, mem->xd->hptr(), &time);

  return true;
}

template <typename T = f4, typename E = u4, bool LENIENT = true>
bool run_gpu(
    MEMPOOL_GPU* mem, double const eb, int const radius, GpuStreamT stream)
{
  float time;

  psz_comp_l23r<T, E, false>(
      mem->od->dptr(), mem->od->template len3<dim3>(), eb, radius,
      mem->ectrl(), mem->compact, &time, stream);
  GpuStreamSync(stream);

  mem->compact->make_host_accessible(stream);
  GpuStreamSync(stream);

  psz::spv_scatter_naive<PROPER_GPU_BACKEND, T, u4>(
      mem->compact_val(), mem->compact_idx(), mem->compact->num_outliers(),
      mem->xd->dptr(), &time, stream);
  psz::spv_scatter_naive<PROPER_GPU_BACKEND, T, u4>(
      mem->compact_val(), mem->compact_idx(), mem->compact->num_outliers(),
      mem->xdtest->dptr(), &time, stream);
  GpuStreamSync(stream);

  psz_decomp_l23<T, E>(
      mem->ectrl(), mem->e->template len3<dim3>(), mem->xd->dptr(), eb, radius,
      mem->xd->dptr(), &time, stream);
  GpuStreamSync(stream);

  size_t first_non_eb = 0;
  bool error_bounded = psz::error_bounded<SEQ, T>(
      mem->xd->control({D2H})->hptr(), mem->od->hptr(), mem->len, eb,
      &first_non_eb);

  return error_bounded;
}

template <typename T = f4, typename E = u4, bool LENIENT = true>
bool testcase(szt x, szt y, szt z, double const eb, int const radius = 512)
{
  using H = u4;

  GpuStreamT stream;
  GpuStreamCreate(&stream);

  auto mem_gpu = new MEMPOOL_GPU(x, radius, y, z);
  mem_gpu->od->control({Malloc, MallocHost});
  mem_gpu->xd->control({Malloc, MallocHost});
  mem_gpu->xdtest->control({Malloc, MallocHost});

  auto mem_cpu = new MEMPOOL_SEQ(x, radius, y, z);
  mem_cpu->od->control({MallocHost});
  mem_cpu->xd->control({MallocHost});
  mem_cpu->xdtest->control({MallocHost});

  create_dummy_input(mem_gpu->od, mem_gpu->len);
  memcpy(
      mem_cpu->od->hptr(), mem_gpu->od->control({D2H})->hptr(),
      mem_cpu->od->bytes());

  run_seq(mem_cpu, eb, radius);
  run_gpu(mem_gpu, eb, radius, stream);

  // compare eq
  mem_gpu->e->control({D2H});
  auto eq_ectrl = std::equal(
      mem_gpu->e->hbegin(), mem_gpu->e->hend(), mem_cpu->e->hbegin());

  for (auto i = 0; i < mem_gpu->e->len(); i++) {
    cout << i << "\t" << mem_gpu->e->hptr(i) << "\t" << mem_cpu->e->hptr(i)
         << endl;
  }

  //   cout << "eq_ectrl: " << eq_ectrl << endl;

  // compare scattered outliers
  // compare num of outliers
  // check if error bounded

  // clean up
  mem_gpu->od->control({Free, FreeHost});
  mem_gpu->xd->control({Free, FreeHost});
  mem_gpu->xdtest->control({Free, FreeHost});
  mem_cpu->od->control({FreeHost});
  mem_cpu->xd->control({FreeHost});
  mem_cpu->xdtest->control({FreeHost});

  return true;
}