/**
 * @file spvo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-24
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "component/spcodec_vec.hh"
#include "rand.hh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>

template <typename T = float>
int f()
{
    T*       a;            // input
    T*       da;           // decoded
    size_t   len = 10000;  //
    uint8_t* file;         // output
    size_t   filelen;      //
    // float    ms;

    cudaMallocManaged(&a, sizeof(T) * len);
    cudaMallocManaged(&da, sizeof(T) * len);
    cudaMallocManaged(&file, sizeof(T) * len);
    cudaMemset(a, 0x0, sizeof(T) * len);
    cudaMemset(da, 0x0, sizeof(T) * len);
    cudaMemset(file, 0x0, sizeof(T) * len);

    // determine nnz
    auto trials = randint(len) / 10;

    for (auto i = 0; i < trials; i++) {
        auto idx = randint(len);
        a[idx]   = randint(INT32_MAX);
    }
    // CPU counting nnz
    auto nnz_ref = 0;
    for (auto i = 0; i < len; i++) {
        if (a[i] != 0) nnz_ref += 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ////////////////////////////////////////////////////////////////

    auto codec = new cusz::SpcodecVec<T>;

    codec->init(len, 1);
    codec->encode(a, len, file, filelen, stream, false);

    std::cout << "filelen: " << filelen << std::endl;
    codec->decode(file, da, stream);

    ////////////////////////////////////////////////////////////////

    bool same = true;

    for (auto i = 0; i < len; i++) {
        if (a[i] != da[i]) {
            same = false;
            break;
        }
    }

    cudaFree(a);
    cudaFree(da);
    // cudaFree(file);

    cudaStreamDestroy(stream);

    if (same)
        return 0;
    else {
        std::cout << "decomp not okay" << std::endl;
        return -1;
    }
}

int main()
{
    auto all_pass = true;
    auto pass     = true;
    for (auto i = 0; i < 10; i++) {
        pass = f() == 0;
        if (not pass) printf("Not passed on %dth trial.\n", i);
        all_pass &= pass;
    }

    if (all_pass)
        return 0;
    else
        return -1;
}