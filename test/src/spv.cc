/**
 * @file spv.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-24
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "kernel/launch_spv.cuh"
#include "rand.hh"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>

template <typename T = float>
int f()
{
    T*        a;            // input
    T*        da;           // decoded
    size_t    len = 10000;  //
    T*        val;          // intermeidate
    uint32_t* idx;          //
    int       nnz;          //
    float     ms;

    hipMallocManaged(&a, sizeof(T) * len);
    hipMallocManaged(&da, sizeof(T) * len);
    hipMallocManaged(&val, sizeof(T) * len);
    hipMallocManaged(&idx, sizeof(uint32_t) * len);

    // determine nnz
    auto trials = randint(len) / 1;

    for (auto i = 0; i < trials; i++) {
        auto idx = randint(len);
        a[idx]   = randint(INT32_MAX);
    }

    // CPU counting nnz
    auto nnz_ref = 0;
    for (auto i = 0; i < len; i++) {
        if (a[i] != 0) nnz_ref += 1;
    }

    hipStream_t stream;
    hipStreamCreate(&stream);

    ////////////////////////////////////////////////////////////////

    launch_spv_gather<T, uint32_t>(a, len, val, idx, nnz, ms, stream);

    hipStreamSynchronize(stream);

    if (nnz != nnz_ref) {
        std::cout << "nnz_ref: " << nnz_ref << std::endl;
        std::cout << "nnz: " << nnz << std::endl;
        std::cerr << "nnz != nnz_ref" << std::endl;
        return -1;
    }

    launch_spv_scatter<T, uint32_t>(val, idx, nnz, da, ms, stream);

    hipStreamSynchronize(stream);

    ////////////////////////////////////////////////////////////////

    bool same = true;

    for (auto i = 0; i < len; i++) {
        if (a[i] != da[i]) {
            same = false;
            break;
        }
    }

    hipFree(a);
    hipFree(da);
    hipFree(val);
    hipFree(idx);

    hipStreamDestroy(stream);

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
