/**
 * @file rand.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-25
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B160F9D0_4352_4049_9B85_57BEAFC5C816
#define B160F9D0_4352_4049_9B85_57BEAFC5C816

#include <stdint.h>
#include <stdlib.h>


namespace psz {
namespace testutils {

namespace cpp {

int randint(size_t upper_limit);

template <typename T>
T randfp(T upper = 1.0, T lower = 0.0);

template <typename T>
void rand_array(T* array, size_t len);

}

namespace cu_hip {

template <typename T>
void rand_array(T* array, size_t len, uint32_t seed = 0x2468);

}

namespace dpcpp {

template <typename T>
void rand_array(T* array, size_t len, uint32_t seed = 0x2468);

}

}  // namespace testutils
}  // namespace psz

#endif /* B160F9D0_4352_4049_9B85_57BEAFC5C816 */
