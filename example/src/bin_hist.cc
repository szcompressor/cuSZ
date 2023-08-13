/**
 * @file bin_hist.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-25
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <string>

#include "hf/hf.hh"
#include "hf/hf_bookg.hh"
#include "hf/hf_codecg.hh"
#include "kernel/histsp.hh"
#include "mem/memseg_cxx.hh"
#include "stat/compare_gpu.hh"
#include "stat/stat.hh"
#include "utils/print_gpu.hh"
#include "utils/viewer.hh"

using T = uint32_t;

int main(int argc, char** argv)
{
  if (argc < 5) {
    printf("PROG /path/to/datafield X Y Z \n");
    printf("0    1                  2 3 4 \n");
    exit(0);
  }
  else {
    auto fname = std::string(argv[1]);
    auto x = atoi(argv[2]);
    auto y = atoi(argv[3]);
    auto z = atoi(argv[4]);

    auto len = x * y * z;
    auto booklen = 1024;

    auto wn = new pszmem_cxx<T>(len, 1, 1, "whole numbers");
    auto freq1 = new pszmem_cxx<uint32_t>(booklen, 1, 1, "frequency");
    auto freq2 = new pszmem_cxx<uint32_t>(booklen, 1, 1, "frequency");

    wn->control({Malloc, MallocHost})
        ->file(fname.c_str(), FromFile)
        ->control({H2D});
    freq1->control({Malloc, MallocHost});
    freq2->control({Malloc, MallocHost});

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float t_hist1, t_hist2;

    psz::stat::histogram<psz_policy::CUDA, T>(
        wn->dptr(), len, freq1->dptr(), booklen, &t_hist1, stream);

    histsp<psz_policy::CUDA, T, uint32_t>(
        wn->dptr(), len, freq1->dptr(), booklen, stream);

    freq1->control({D2H});
    freq2->control({D2H});

    for (auto i = 0; i < booklen; i++) {
      printf(
          "idx:\t%d\t"
          "hist:\t%u\t"
          "histsp:\t%u\t"
          "\n",
          i, freq1->hptr(i), freq2->hptr(i));
    }

    delete wn;
    delete freq1;
    delete freq2;

    cudaStreamDestroy(stream);
  }

  return 0;
}