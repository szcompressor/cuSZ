/**
 * @file analysis.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-11-07
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef ANALYSIS_CUH
#define ANALYSIS_CUH

#include <tuple>

template <typename Data>
std::tuple<Data, Data, Data, Data> GetMinMaxRng(thrust::device_ptr<Data> g_ptr, size_t len);

template <typename Data>
void GetPSNR(Data* x, Data* y, size_t len);

#endif
