/**
 * @file types.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2019-06-08
 *
 * @todo separate type definition and cuSZ configuration for driver program (header).
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef TYPES_HH
#define TYPES_HH

#include <algorithm>
#include <cmath>    // for FP32 bit representation
#include <cstddef>  // size_t
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "format.hh"
#include "io.hh"
#include "timer.hh"

using namespace std;

template <typename T>
double GetDatumValueRange(string fname, size_t l);

size_t* InitializeDims(size_t cap, size_t n_dims, size_t dim0, size_t dim1 = 1, size_t dim2 = 1, size_t dim3 = 1);

void SetDims(size_t* dims_L16, size_t new_dims[4]);

typedef struct ErrorBoundConfigurator {
    int         capacity, radius;
    double      base, exp_base2, exp_base10;
    double      eb_base2, eb_base10, eb_final;
    std::string mode;

    void ChangeToRelativeMode(double value_range);

    void ChangeToTightBase2();

    ErrorBoundConfigurator(int _capacity = 32768, double _precision = 1, double _exponent = -3, int _base = 10);

    void debug() const;

} config_t;

// typedef struct DimensionInfo          dim_t;
// typedef struct ErrorBoundConfigurator config_t;

double* InitializeErrorBoundFamily(struct ErrorBoundConfigurator* eb_config);

/*
2^-1  2^-2  2^-3
10^-1
2^-4  2^-5  2^-6
10^-2
2^-7  2^-8  2^-9
10^-3
2^-10 2^-11 2^-12 2^-13
10^-4
2^-14 2^-15 2^-16
10^-5
2^-17 2^-18 2^-19
10^-6
2^-20 2^-21 2^-22 2^-23
10^-7
2^-24 2^-25 2^-26
10^-8
2^-27 2^-28 2^-29
10^-9
2^-30 2^-31 2^-32 2^-33
10^-10
2^-34
 */

// static std::unordered_map<int8_t, int8_t> exp_dec2bin = {{-1, -4},  {-2, -7},  {-3, -10}, {-4, -14}, {-5, -17},
//                                                         {-6, -20}, {-7, -24}, {-8, -27}, {-9, -30}, {-10, -34}};

#endif /* TYPES_HH */
