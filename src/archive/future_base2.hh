#ifndef BASE2_HH
#define BASE2_HH

/**
 * @file future_base2.hh
 * @author Jiannan Tian
 * @brief Internal use: base2 operations.
 * @version 0.1
 * @date 2020-09-20
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdlib>

namespace prototype {
auto __div64by = [](double& __n, uint64_t const& __p) { reinterpret_cast<uint64_t&>(__n) -= (__p << 52u); };
auto __div32by = [](float& __n, uint32_t const& __p) { reinterpret_cast<uint32_t&>(__n) -= (__p << 23u); };
}  // namespace prototype

#endif
