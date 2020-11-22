/**
 * @file analysis_utils.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-11-04
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "analysis_utils.hh"
#include <iostream>
#include <tuple>

// not working for now
template <typename T>
double GetDatumValueRange(std::string fname, size_t l)
{
    auto d = io::ReadBinaryFile<T>(fname, l);

    int idx;
    int max_val, min_val; /* = 0 not needed according to Jim Cownie comment */

#pragma omp parallel for reduction(max : max_val)
    for (idx = 0; idx < l; idx++) max_val = max_val > d[idx] ? max_val : d[idx];
#pragma omp parallel for reduction(min : min_val)
    for (idx = 0; idx < l; idx++) min_val = min_val < d[idx] ? min_val : d[idx];

    delete[] d;
    return max_val - min_val;
}

template double GetDatumValueRange<float>(std::string fname, size_t l);
template double GetDatumValueRange<double>(std::string fname, size_t l);

template <typename T>
std::tuple<double, double, double> GetDatumValueRange(T* data, size_t l)
{
    int idx;
    T   max_val = data[0], min_val = data[0]; /* = 0 not needed according to Jim Cownie comment */

    // #pragma omp parallel for reduction(max : max_val) reduction(min : min_val)
    for (idx = 0; idx < l; idx++) {
        max_val = max_val > data[idx] ? max_val : data[idx];
        min_val = min_val < data[idx] ? min_val : data[idx];
    }
    // std::cout << max_val << std::endl;
    // std::cout << min_val << std::endl;

    return {max_val, min_val, max_val - min_val};
}

template std::tuple<double, double, double> GetDatumValueRange<float>(float*, size_t);
template std::tuple<double, double, double> GetDatumValueRange<double>(double*, size_t);