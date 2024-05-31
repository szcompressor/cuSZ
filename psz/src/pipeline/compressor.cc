/**
 * @file compressor.cc
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2023-06-02
 * (create) 2020-02-12
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2023 by Indiana University
 * See LICENSE in top-level directory
 *
 */

#include "compressor.hh"

#include "pipeline/compressor.inl"
#include "port.hh"
#include "tehm.hh"

using CompoundF4 = psz::CompoundType<float>;
using CF4 = psz::Compressor<CompoundF4>;
template class psz::Compressor<CompoundF4>;
template CF4* CF4::init<psz_context>(psz_context*, bool, bool);
template CF4* CF4::init<psz_header>(psz_header*, bool, bool);

using CompoundF8 = psz::CompoundType<double>;
using CF8 = psz::Compressor<CompoundF8>;
template class psz::Compressor<CompoundF8>;
template CF8* CF8::init<psz_context>(psz_context*, bool, bool);
template CF8* CF8::init<psz_header>(psz_header*, bool, bool);
