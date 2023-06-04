/**
 * @file definition.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-20
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMMON_DEFINITION_HH
#define CUSZ_COMMON_DEFINITION_HH

#include <cstdint>
#include <tuple>
#include <vector>

namespace cusz {

// using FREQ = uint32_t;

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord      = std::vector<TimeRecordTuple>;
using timerecord_t    = TimeRecord*;

// using BYTE = uint8_t;

};  // namespace cusz

#endif
