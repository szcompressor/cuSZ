/**
 * @file demo_fp_int.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-07
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace std;

typedef struct MultiByte4 {
    unsigned char _1 : 8;
    unsigned char _2 : 8;
    unsigned char _3 : 8;
    unsigned char _4 : 8;
} _byte4;

template <size_t n, size_t m>
struct _FP {
    enum { mantissa_bits = m };
    enum { exponent_bits = n };
    unsigned int mantissa : m;
    unsigned int exponent : n;
    unsigned int signum : 1;
};

template <size_t n, size_t m>
struct _FP_2parts {
    enum { mantissa_bits = m };
    enum { exponent_bits = n };
    unsigned int mantissa_low : 8;
    unsigned int mantissa_mid : 8;
    unsigned int mantissa_high : m - 16;
    unsigned int exponent : n;
    unsigned int signum : 1;
};

using FP   = struct _FP<8, 23>;
using FP3p = struct _FP_2parts<8, 23>;

unsigned int mantissa_value(FP f)
{  //
    return (1 << (f.exponent - 127)) + (f.mantissa >> (23 - (f.exponent - 127)));
}

void print_sample()
{
    for (auto i = 511; i <= 512; i++) {
        float a = i;
        auto  b = *reinterpret_cast<unsigned int*>(&a);
        // cout << bitset<32>(b) << '\t';
        cout << bitset<8>(b >> 24) << ' ';
        cout << bitset<8>(b >> 16) << ' ';
        cout << bitset<8>(b >> 8) << ' ';
        cout << bitset<8>(b) << '\t';

        auto c = *reinterpret_cast<FP*>(&a);
        cout << "number: " << i << "\t";
        cout << "exponent: " << c.exponent << '\t' << bitset<c.exponent_bits>(c.exponent) << "  ";
        cout << "E-127: " << c.exponent - 127 << "\t";
        auto mval = mantissa_value(c);
        cout << "mantissa: " << bitset<c.mantissa_bits>(c.mantissa) << "  ";
        cout << "mantissa val: " << bitset<c.mantissa_bits>(mval) << " (" << mval << ")\n";
    }
}

void test_fpint_hist(int num)
{
    cout << "num: " << num << "\n";
    // histogram_fp_represented_int
    vector<unsigned int> hist_fpint(256, 0);

    for (auto i = 0; i < num; i++) {
        float a = i;
        auto  d = *reinterpret_cast<_byte4*>(&a);
        hist_fpint[d._1] += 1;
        hist_fpint[d._2] += 1;
        hist_fpint[d._3] += 1;
        hist_fpint[d._4] += 1;
    }

    sort(hist_fpint.begin(), hist_fpint.end(), [](int a, int b) { return a > b; });

    for (auto i = 0; i < 256; i++) { cout << i << '\t' << hist_fpint[i] << '\n'; }

    double entropy = 0;
    for (auto& i : hist_fpint) {
        if (i != 0) {
            auto p = 1.0 * i / (num * 4);
            entropy += -log2(p) * p;
        }
    }
    cout << "entropy: " << entropy << '\t';
    cout << "p1: " << hist_fpint[0] * 1.0 / (num * 4) << endl;
}

int main()
{
    print_sample();
    // test_fpint_hist(10);
    // test_fpint_hist(100);
    // test_fpint_hist(1000);
    // test_fpint_hist(10000);
//    test_fpint_hist(100000);
    // test_fpint_hist(1000000);
    return 0;
}