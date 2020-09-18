//
//  types.hh
//  waveSZ
//
//  Created by JianNan Tian on 6/8/19.
//  Copyright © 2019 JianNan Tian. All rights reserved.
//

#include <algorithm>
#include <cmath>    // for FP32 bit representation
#include <cstddef>  // size_t
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "constants.hh"
#include "format.hh"
#include "io.hh"
#include "timer.hh"
#include "types.hh"

using namespace std;

template <typename T>
double GetDatumValueRange(string fname, size_t l)
{
    auto d    = io::ReadBinaryFile<T>(fname, l);
    T    max_ = *std::max_element(d, d + l);
    T    min_ = *std::min_element(d, d + l);
    delete[] d;
    return max_ - min_;
}

template double GetDatumValueRange<float>(string fname, size_t l);
template double GetDatumValueRange<double>(string fname, size_t l);

size_t* InitializeDims(size_t cap, size_t n_dims, size_t dim0, size_t dim1, size_t dim2, size_t dim3)
{
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
void SetDims(size_t* dims_L16, size_t new_dims[4])
{
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

// typedef struct ErrorBoundConfigurator {
//    int         capacity, radius;
//    double      base, exp_base2, exp_base10;
//    double      eb_base2, eb_base10, eb_final;
//    std::string mode;
ErrorBoundConfigurator::ErrorBoundConfigurator(int _capacity, double _precision, double _exponent, int _base)
{
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

    cout << log_info << "quant.capacity:\t" << _capacity << endl;
    if (_base == 10) {
        cout << log_info << "input eb:\t" << _precision;
        cout << " x 10^(" << _exponent << ") = " << eb_final << endl;
    }
    else if (_base == 2) {
        cout << "eb.set.to:\t"
             << "2^(" << _exponent << ") = " << eb_final << endl;
    }
}

void ErrorBoundConfigurator::ChangeToRelativeMode(double value_range)
{
    if (value_range == 0) {
        cerr << log_err << "INVALID VALUE RANGE!" << endl;
        exit(1);
    }
    // cout << log_info << "change to r2r mode \e[2m(relative-to-value-range)\e[0m" << endl;
    cout << log_info << "eb change:\t" << eb_final << " (input eb) x " << value_range << " (rng) = ";
    this->eb_final *= value_range;
    cout << eb_final;
    cout << " \e[2m(relative-to-value-range, r2r mode)\e[0m" << endl;
    mode = std::string("VRREL");
}

void ErrorBoundConfigurator::ChangeToTightBase2()
{
    base = 2;
    cout << log_info << "switch.to.tight.base2.mode, eb changed from " << eb_final << " = 2^(" << exp_base2 << ") to ";
    cout << "the exp base2 before changing:\t" << exp_base2 << endl;
    exp_base2 = floor(exp_base2);
    cout << "the exp base2 after changing:\t" << exp_base2 << endl;
    eb_final = pow(2, exp_base2);
    cout << eb_final << " = 2^(" << exp_base2 << ")" << endl;
}

void ErrorBoundConfigurator::debug() const
{
    cout << log_dbg;
    printf("exponent = %.3f (base10) (or) %.3f (base2)\n", exp_base10, exp_base2);
}

//} config_t;

typedef struct ErrorBoundConfigurator config_t;

double* InitializeErrorBoundFamily(config_t* eb_config)
{
    auto ebs_L4 = new double[4]();
    ebs_L4[0]   = eb_config->eb_final;            // eb
    ebs_L4[1]   = 1 / eb_config->eb_final;        // 1/eb
    ebs_L4[2]   = 2 * eb_config->eb_final;        // 2* eb
    ebs_L4[3]   = 1 / (2 * eb_config->eb_final);  // 1/(2*eb)
    return ebs_L4;
}
