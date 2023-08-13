/**
 * @file base_compressor.cuh
 * @author Jiannan Tian
 * @brief T-only Base Compressor; can also be used for dryrun.
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef BB504423_E0DF_4AAA_8AF3_BEEEA28053DB
#define BB504423_E0DF_4AAA_8AF3_BEEEA28053DB

#include "context.h"
#include "kernel/dryrun.cuh"
#include "mem/memseg_cxx.hh"
#include "stat/compare_gpu.hh"
#include "type_traits.hh"
#include "utils/analyzer.hh"
#include "utils/config.hh"
#include "utils/cuda_err.cuh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/verify.hh"
#include "utils/viewer.hh"

namespace cusz {

template <typename T>
Dryrunner<T>& Dryrunner<T>::generic_dryrun(
    const std::string fname, double eb, int radius, bool r2r,
    cudaStream_t stream)
{
  throw std::runtime_error("Generic dryrun is disabled.");
  return *this;
}

template <typename T>
Dryrunner<T>& Dryrunner<T>::dualquant_dryrun(
    const std::string fname, double eb, bool r2r, cudaStream_t stream)
{
  // auto len = original.len();
  auto len = original->len();

  original->debug();

  original->file(fname.c_str(), FromFile)->control({ASYNC_H2D}, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (r2r) original->extrema_scan(max, min, rng), eb *= rng;

  auto ebx2_r = 1 / (eb * 2);
  auto ebx2 = eb * 2;

  cusz::dualquant_dryrun_kernel                                           //
      <<<psz_utils::get_npart(len, 256), 256, 256 * sizeof(T), stream>>>  //
      (original->dptr(), reconst->dptr(), len, ebx2_r, ebx2);

  reconst->control({ASYNC_D2H}, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cusz_stats stat;
  psz::thrustgpu_assess_quality(&stat, reconst->hptr(), original->hptr(), len);
  psz::print_metrics_cross<T>(&stat, 0, true);

  return *this;
}

template <typename T>
Dryrunner<T>::~Dryrunner()
{
  delete original;
  delete reconst;
}

template <typename T>
Dryrunner<T>& Dryrunner<T>::init_generic_dryrun(dim3 size)
{
  throw std::runtime_error("Generic dryrun is disabled.");
  return *this;
}

template <typename T>
Dryrunner<T>& Dryrunner<T>::destroy_generic_dryrun()
{
  throw std::runtime_error("Generic dryrun is disabled.");
  return *this;
}

template <typename T>
Dryrunner<T>& Dryrunner<T>::init_dualquant_dryrun(dim3 size)
{
  original = new pszmem_cxx<T>(size.x, size.y, size.z, "original");
  original->control({MallocHost, Malloc});

  reconst = new pszmem_cxx<T>(size.x, size.y, size.z, "reconst");
  reconst->control({MallocHost, Malloc});

  return *this;
}

template <typename T>
Dryrunner<T>& Dryrunner<T>::destroy_dualquant_dryrun()
{
  original->control({FreeHost, Free});
  reconst->control({FreeHost, Free});

  return *this;
}

}  // namespace cusz

#endif /* BB504423_E0DF_4AAA_8AF3_BEEEA28053DB */
