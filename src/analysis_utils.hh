/**
 * @file analysis_utils.hh
 * @author Jiannan Tian
 * @brief Simple data analysis (header)
 * @version 0.1.3
 * @date 2020-11-03
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef ANALYSIS_UTILS_H
#define ANALYSIS_UTILS_H

#include <omp.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include "io.hh"

template <typename T>
double GetDatumValueRange(std::string fname, size_t l);

template <typename T>
std::tuple<double, double, double> GetDatumValueRange(T* data, size_t l);

namespace cusz {
namespace impl {

inline size_t GetEdgeOfReinterpretedSquare(size_t l) { return static_cast<size_t>(ceil(sqrt(l))); };

}  // namespace impl
}  // namespace cusz

// TODO replace with mirrored data w/ metadata
template <typename T>
struct DataPack {
    T*     data;
    T*     d_data;
    size_t len;
    size_t m{}, mxm{};  // m is the smallest possible integer that is larger than sqrt(len)

    DataPack(T* data_, T* d_data_, size_t len_)
    {
        data      = data_;
        d_data    = d_data_;
        len       = len_;
        this->m   = cusz::impl::GetEdgeOfReinterpretedSquare(len);  // row-major mxn matrix
        this->mxm = m * m;
    }
};

#endif