/**
 * @file test_spgs.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-02
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include "../src/common/capsule.hh"
#include "../src/wrapper/spgs.cuh"

using std::cout;

template <typename T>
T* allocate(unsigned int len)
{
    T* a;
    cudaMallocManaged(&a, len * sizeof(T));
    cudaMemset(a, 0x00, len * sizeof(T));
    return a;
}

void f()
{
    constexpr auto UNIFIED = false;
    constexpr auto LOC     = cusz::LOC::HOST_DEVICE;

    // constexpr auto UNIFIED = true;
    // constexpr auto LOC     = cusz::LOC::UNIFIED;

    auto len = 512 * 512 * 512;
    int  out_nnz;
    auto bytes = len * sizeof(float);

    Capsule<float, UNIFIED> data(len);
    Capsule<int, UNIFIED>   sp_idx(len / 20);
    Capsule<float, UNIFIED> sp_val(len / 20);

    data.template alloc<LOC>();
    sp_val.template alloc<LOC>();
    sp_idx.template alloc<LOC>();

    data.hptr[20] = 21;
    data.hptr[30] = 22;
    data.hptr[40] = 23;
    data.hptr[50] = 24;

    data.host2device();

    cusz::spGS<float> spreducer;

    float        ms;
    unsigned int dump_nbyte;

    spreducer.gather(data.dptr, len, nullptr, sp_idx.dptr, sp_val.dptr, out_nnz, dump_nbyte);
    ms = spreducer.get_time_elapsed();

    sp_val.device2host();
    sp_idx.device2host();

    cout << (bytes * 1.0 / 1e9) / (ms / 1e3) << " GiB/s\n";

    cout << "nnz: " << out_nnz << '\n';
    for (auto i = 0; i < out_nnz; i++)  //
        cout << sp_idx.hptr[i] << '\t' << sp_val.hptr[i] << '\n';

    data.template free<LOC>();

    ////////////////////////////////////////////////////////////////////////////////

    data.template alloc<LOC>();

    spreducer.scatter(sp_idx.dptr, sp_val.dptr, out_nnz, data.dptr);

    data.device2host();

    ms = spreducer.get_time_elapsed();
    cout << (bytes * 1.0 / 1e9) / (ms / 1e3) << " GiB/s\n";

    cout << data.hptr[20] << '\n';
    cout << data.hptr[30] << '\n';
    cout << data.hptr[40] << '\n';
    cout << data.hptr[50] << '\n';

    data.template free<LOC>();
    sp_val.template free<LOC>();
    sp_idx.template free<LOC>();
}

int main()
{
    f();
    return 0;
}