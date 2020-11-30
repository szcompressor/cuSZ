/**
 * @file analysis_utils.hh
 * @author Jiannan Tian
 * @brief Simple data analysis (header)
 * @version 0.1.3
 * @date 2020-11-03
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
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

template <typename T>
struct AdHocDataPack {
    T*     data;
    T*     d_data;
    size_t len;
    size_t m;    // padded len, single dimension
    size_t mxm;  // 2d

    AdHocDataPack(T* data_, T* d_data_, size_t len_)
    {
        data      = data_;
        d_data    = d_data_;
        len       = len_;
        this->m   = cusz::impl::GetEdgeOfReinterpretedSquare(len);  // row-major mxn matrix
        this->mxm = m * m;
    }
};

// namespace cusz {
// namespace analysis {

// template <typename T>
// double GetDatumRange(T* d_data);

// }  // namespace analysis
// }  // namespace cusz

#endif