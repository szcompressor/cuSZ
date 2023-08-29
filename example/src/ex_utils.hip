/**
 * @file ex_utils.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-13
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

// deps
#include "port.hh"
#include "cusz/type.h"
// definitions
#include "ex_utils_g.inl"

template u4 count_outlier(float*, size_t, int, void*);
template u4 count_outlier(u1*, size_t, int, void*);
template u4 count_outlier(u2*, size_t, int, void*);
template u4 count_outlier(u4*, size_t, int, void*);

#ifdef __MAIN__

#include "mem/memseg_cxx.hh"

int main()
{
  auto radius = 512;
  auto a = new pszmem_cxx<float>(100, 100, 1);
  a->control({Malloc, MallocHost});

  auto _count = 0;
  for (auto i = 0; i < 100 * 99; i += 100) a->hptr(i) = 2045, _count += 1;

  a->control({H2D});

  auto num = count_outlier(a->dptr(), a->len(), radius, nullptr);

  printf("supposed:\t%lu\n", _count);
  printf("counted:\t%lu\n", num);

  delete a;

  return 0;
}

#endif