/**
 * @file print_arr.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AB8F2CD4_0C03_41C5_8FB0_0923DA702486
#define AB8F2CD4_0C03_41C5_8FB0_0923DA702486

#include "busyheader.hh"
#include "cusz/type.h"

namespace psz {

template <typename T>
void peek_data(T* h_arr, size_t num)
{
  int counter = 0;
  if (std::numeric_limits<T>::is_integer) {
    if (std::numeric_limits<T>::is_signed) {
      std::for_each(h_arr, h_arr + num, [&](T& n) {
        printf("%6ld", (i8)n);
        if (counter % 10 == 0 and counter != 0) printf("\n");
        counter++;
      });
    }
    else {
      std::for_each(h_arr, h_arr + num, [&](T& n) {
        printf("%6lu", (u8)n);
        if (counter % 10 == 0 and counter != 0) printf("\n");
        counter++;
      });
    }
  }
  else if (std::is_floating_point<T>::value) {
    int counter = 0;
    std::for_each(h_arr, h_arr + num, [&](T& n) {
      printf("%10.6lf", (double)n);
      if (counter % 10 == 0 and counter != 0) printf("\n");
      counter++;
    });
  }
  else {
    std::runtime_error("peek_data cannot accept this type.");
  }
}

}  // namespace psz



#endif /* AB8F2CD4_0C03_41C5_8FB0_0923DA702486 */
