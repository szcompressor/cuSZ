/**
 * @file bin_viadisk.cc
 * @author Jiannan Tian
 * @brief Disaggregated for cross-platform demonstration.
 * @version 0.4
 * @date 2023-06-12
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernel/l23.hh"
#include "kernel/lproto.hh"
#include "kernel2/l23r.hh"
#include "pipeline/compact_cuda.inl"
#include "utils2/memseg_cxx.hh"

using std::string;

using T = float;
using E = uint32_t;
using H = uint32_t;
using FP = T;

void predict(
    char const* ifn, size_t const x, size_t const y, size_t const z,
    double const eb = 1e-4, int const radius = 128, int version = 0)
{
  auto len = x * y * z;
  auto len3 = dim3(x, y, z);
  auto oridata = new pszmem_cxx<T>(x, y, z, "oridata");
  auto errctrl = new pszmem_cxx<E>(x, y, z, "errctrl");
  auto outlier = new pszmem_cxx<T>(x, y, z, "outlier");

  oridata->control({Malloc, MallocHost})->file(ifn, FromFile)->control({H2D});
  errctrl->control({Malloc, MallocHost});
  outlier->control({Malloc, MallocHost});

  CompactCudaDram<T> compact;
  compact.reserve_space(len).malloc().mallochost();

  float time;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  if (version == -1) {
    psz_comp_lproto<T, E>(
        oridata->dptr(), len3, eb, radius, errctrl->dptr(), outlier->dptr(),
        &time, stream);
  }
  else if (version == 0) {
    psz_comp_l23<T, E, FP>(
        oridata->dptr(), len3, eb, radius, errctrl->dptr(), outlier->dptr(),
        &time, stream);
  }
  else if (version == 1) {
    psz_comp_l23r<T>(
        oridata->dptr(), len3, eb, radius, errctrl->dptr(), outlier->dptr(),
        &time, stream);
  }
  else {
    throw std::runtime_error("no such verison for `predict`");
  }

  delete oridata;
  delete errctrl;
  delete outlier;

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  auto x = atoi(argv[2]);
  auto y = atoi(argv[3]);
  auto z = atoi(argv[4]);

  predict(argv[1], x, y, z, 1e-4, 128, 0);
  return 0;
}
