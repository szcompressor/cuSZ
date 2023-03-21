/**
 * @file test_l23_scan.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-23
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;

#include "lorenzo.inl"
#include "lorenzo23.inl"
#include "rand.hh"

template <int BLOCK = 256, int SEQ = 8>
bool test_inclusive_scan()
{
    using T  = float;
    using EQ = int32_t;
    using FP = T;

    constexpr auto NTHREAD = BLOCK / SEQ;

    auto len  = BLOCK;
    auto ebx2 = 1;

    T*  data{nullptr};
    EQ* eq{nullptr};

    cudaMallocManaged(&data, sizeof(T) * len);
    cudaMallocManaged(&eq, sizeof(EQ) * len);
    cudaMemset(eq, 0x0, sizeof(EQ) * len);

    bool ok = true;

    {
        cout << "original" << endl;
        for (auto i = 0; i < BLOCK; i++) data[i] = 1;

        cusz::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<1, NTHREAD>>>(data, eq, data, dim3(len, 1, 1), dim3(0, 0, 0), 0, ebx2);
        cudaDeviceSynchronize();

        // for (auto i = 0; i < BLOCK; i++) { cout << data[i] << " "; }
        // cout << "\n" << endl;
        for (auto i = 0; i < BLOCK; i++)
            if (data[i] != i) {
                ok = false;
                break;
            }

        if (not ok) {
            printf("ERROR\toriginal (21-May) not correct\n");
            return false;
        }
    }

    {
        cout << "refactored v0 (wave32)" << endl;
        for (auto i = 0; i < BLOCK; i++) data[i] = 1;

        parsz::cuda::__kernel::v0::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<1, NTHREAD>>>(eq, data, dim3(len, 1, 1), dim3(0, 0, 0), 0, ebx2, data);
        cudaDeviceSynchronize();

        // for (auto i = 0; i < BLOCK; i++) cout << data[i] << " ";
        // cout << "\n" << endl;
        for (auto i = 0; i < BLOCK; i++)
            if (data[i] != i) {
                ok = false;
                break;
            }

        if (not ok) {
            printf("ERROR\trefactored (22-Dec) not correct\n");
            return false;
        }
    }

    cudaFree(data);
    cudaFree(eq);

    return ok;
}

int main()
{
    auto all_pass = true;
    all_pass      = all_pass and test_inclusive_scan<256, 4>();
    all_pass      = all_pass and test_inclusive_scan<256, 8>();
    all_pass      = all_pass and test_inclusive_scan<512, 4>();
    all_pass      = all_pass and test_inclusive_scan<512, 8>();
    all_pass      = all_pass and test_inclusive_scan<1024, 4>();
    all_pass      = all_pass and test_inclusive_scan<1024, 8>();

    if (all_pass)
        return 0;
    else
        return -1;
}