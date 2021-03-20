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
#include "utils/io.hh"

template <typename T>
double GetDatumValueRange(std::string fname, size_t l);

template <typename T>
std::tuple<double, double, double> GetDatumValueRange(T* data, size_t l);

namespace cusz {
namespace impl {

inline size_t GetEdgeOfReinterpretedSquare(size_t l) { return static_cast<size_t>(ceil(sqrt(l))); };

}  // namespace impl
}  // namespace cusz

// TODO move elsewhere
template <typename T>
class DataPack {
   public:
    T*     h;
    T*     d;
    size_t len{}, sqrt_ceil{}, pseudo_matrix_size{};

    DataPack() = default;

    DataPack<T>& SetHostSpace(T* h_)
    {
        h = h_;
        return *this;
    }
    DataPack<T>& SetDeviceSpace(T* d_)
    {
        d = d_;
        return *this;
    }
    DataPack<T>& SetLen(size_t len_)
    {
        this->len                = len_;
        this->sqrt_ceil          = static_cast<size_t>(ceil(sqrt(len)));  // row-major mxn matrix
        this->pseudo_matrix_size = sqrt_ceil * sqrt_ceil;
        return *this;
    }
};

#endif