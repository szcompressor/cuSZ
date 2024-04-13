/**
 * @file vis_stat.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-25
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E4F18FA1_7CF5_4292_B8D2_D009959C72D9
#define E4F18FA1_7CF5_4292_B8D2_D009959C72D9

#include <cstddef>
#include <string>

template <typename T>
double get_entropy(T* code, size_t l, size_t cap = 1024);

template <typename T>
void visualize_histogram(
    const std::string& tag, T* _d_POD, size_t l, size_t _bins = 16,
    bool log_freq = false, double override_min = 0, double override_max = 0,
    bool eliminate_zeros = false, bool use_scientific_notation = true);

#endif /* E4F18FA1_7CF5_4292_B8D2_D009959C72D9 */
