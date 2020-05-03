//
//  types.hh
//  waveSZ
//
//  Created by JianNan Tian on 6/8/19.
//  Copyright © 2019 JianNan Tian. All rights reserved.
//

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

#include "__io.hh"
#include "format.hh"
#include "timer.hh"

using namespace std;

// TODO standalone metadata management
extern const size_t DIM0   = 0;
extern const size_t DIM1   = 1;
extern const size_t DIM2   = 2;
extern const size_t DIM3   = 3;
extern const size_t nBLK0  = 4;
extern const size_t nBLK1  = 5;
extern const size_t nBLK2  = 6;
extern const size_t nBLK3  = 7;
extern const size_t nDIM   = 8;
extern const size_t LEN    = 12;
extern const size_t CAP    = 13;
extern const size_t RADIUS = 14;

extern const size_t EB     = 0;
extern const size_t EBr    = 1;
extern const size_t EBx2   = 2;
extern const size_t EBx2_r = 3;

extern const int B_1d = 32;
extern const int B_2d = 16;
extern const int B_3d = 8;

#if !defined(__APPLE__)  // ISSUE for HLS compilation on Windows
typedef uint32_t u_int32_t;
typedef uint64_t u_int64_t;
#endif

size_t const DEFAULT_VAL = 1;

template <typename T>
double GetDatumValueRange(string fname, size_t l) {
    auto d    = io::ReadBinaryFile<T>(fname, l);
    T    max_ = *std::max_element(d, d + l);
    T    min_ = *std::min_element(d, d + l);
    delete[] d;
    return max_ - min_;
}

size_t* InitializeDims(size_t cap,  //
                       size_t n_dims,
                       size_t dim0,
                       size_t dim1 = 1,
                       size_t dim2 = 1,
                       size_t dim3 = 1) {
    auto dims_L16 = new size_t[16]();

    size_t dims[] = {dim0, dim1, dim2, dim3};
    std::copy(dims, dims + 4, dims_L16);
    dims_L16[nDIM] = n_dims;

    int BLK;
    if (dims_L16[nDIM] == 1)
        BLK = B_1d;
    else if (dims_L16[nDIM] == 2)
        BLK = B_2d;
    else if (dims_L16[nDIM] == 3)
        BLK = B_3d;

    dims_L16[nBLK0]  = (dims_L16[DIM0] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK1]  = (dims_L16[DIM1] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK2]  = (dims_L16[DIM2] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK3]  = (dims_L16[DIM3] - 1) / (size_t)BLK + 1;
    dims_L16[LEN]    = dims_L16[DIM0] * dims_L16[DIM1] * dims_L16[DIM2] * dims_L16[DIM3];
    dims_L16[CAP]    = cap;
    dims_L16[RADIUS] = cap / 2;

    return dims_L16;
}

// for example, binning needs to set new dimensions
void SetDims(size_t* dims_L16, size_t new_dims[4]) {
    std::copy(new_dims, new_dims + 4, dims_L16);
    int BLK;
    if (dims_L16[nDIM] == 1)
        BLK = B_1d;
    else if (dims_L16[nDIM] == 2)
        BLK = B_2d;
    else if (dims_L16[nDIM] == 3)
        BLK = B_3d;
    dims_L16[nBLK0] = (dims_L16[DIM0] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK1] = (dims_L16[DIM1] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK2] = (dims_L16[DIM2] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK3] = (dims_L16[DIM3] - 1) / (size_t)BLK + 1;
    dims_L16[LEN]   = dims_L16[DIM0] * dims_L16[DIM1] * dims_L16[DIM2] * dims_L16[DIM3];
}

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

static std::unordered_map<int8_t, int8_t> exp_dec2bin = {{-1, -4},  {-2, -7},  {-3, -10}, {-4, -14}, {-5, -17},
                                                         {-6, -20}, {-7, -24}, {-8, -27}, {-9, -30}, {-10, -34}};

typedef struct ErrorBoundConfigurator {
    int         capacity, radius;
    double      base, exp_base2, exp_base10;
    double      eb_base2, eb_base10, eb_final;
    std::string mode;

    void ChangeToRelativeMode(double value_range) {
        if (value_range == 0) {
            cerr << log_err << "INVALID VALUE RANGE!" << endl;
            exit(1);
        }
        cout << log_info << "change to r2r mode \e[2m(relative-to-value-range)\e[0m" << endl;
        cout << log_null << "eb --> " << eb_final << " x " << value_range << " = ";
        this->eb_final *= value_range;
        cout << eb_final << endl;
        mode = std::string("VRREL");
    }

    void ChangeToTightBase2() {
        base = 2;
        cout << log_info << "switch.to.tight.base2.mode, eb changed from " << eb_final << " = 2^(" << exp_base2 << ") to ";
        cout << "the exp base2 before changing:\t" << exp_base2 << endl;
        exp_base2 = floor(exp_base2);
        cout << "the exp base2 after changing:\t" << exp_base2 << endl;
        eb_final = pow(2, exp_base2);
        cout << eb_final << " = 2^(" << exp_base2 << ")" << endl;
    }

    ErrorBoundConfigurator(int _capacity = 32768, double _precision = 1, double _exponent = -3, int _base = 10) {
        capacity = _capacity;
        radius   = capacity / 2;
        mode     = std::string("ABS");

        if (_precision != 1 and _base == 2) {
            cerr << "tmp.ly we only support 1 x pow(2, \?\?)" << endl;
        }
        eb_final   = _precision * pow(_base, _exponent);
        base       = _base;
        exp_base10 = _base == 10 ? _exponent : log10(eb_final);
        exp_base2  = _base == 2 ? _exponent : log2(eb_final);

        cout << log_info << "bin.cap:\t\t" << _capacity << endl;
        if (_base == 10) {
            cout << log_info << "user-set eb:\t" << _precision;
            cout << " x 10^(" << _exponent << ") = " << eb_final << endl;
        } else if (_base == 2) {
            cout << "eb.set.to:\t"
                 << "2^(" << _exponent << ") = " << eb_final << endl;
        }
    }

    void debug() {
        cout << log_dbg << "exp.base10:\t" << exp_base10 << endl;
        cout << log_dbg << "exp.base2:\t" << exp_base2 << endl;
        cout << log_dbg << "final.eb:\t" << eb_final << endl;
    }

} config_t;

typedef struct DimensionInfo          dim_t;
typedef struct ErrorBoundConfigurator config_t;
typedef size_t                        psegSize_t;

// TODO move somewhere else
double* InitializeErrorBoundFamily(config_t* eb_config) {
    auto ebs_L4 = new double[4]();
    ebs_L4[0]   = eb_config->eb_final;            // eb
    ebs_L4[1]   = 1 / eb_config->eb_final;        // 1/eb
    ebs_L4[2]   = 2 * eb_config->eb_final;        // 2* eb
    ebs_L4[3]   = 1 / (2 * eb_config->eb_final);  // 1/(2*eb)
    return ebs_L4;
}

#endif /* TYPES_HH */
