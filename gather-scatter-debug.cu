#include <iostream>
#include <string>
#include "src/cuda_mem.cuh"
#include "src/cusz_workflow.cuh"
#include "src/gather_scatter.cuh"
#include "src/io.hh"

using namespace std;

int main(int argc, char** argv)
{
    string fi(argv[1]);
    size_t len = atoi(argv[2]);
    auto   m   = ::cusz::impl::GetEdgeOfReinterpretedSquare(len);

    auto data = new float[m * m];
    io::ReadBinaryFile<float>(fi, data, len);

    // for (auto i = 0; i < 100; i++) {
    //     cout << data[i] << endl;
    // }
    for (auto i = 0; i < len; i++) {
        if (i % 2 == 0) data[i] = 0;
        if (i % 3 == 0) data[i] = 0;
        if (i % 5 == 0) data[i] = 0;
        if (i % 7 == 0) data[i] = 0;
        if (i % 11 == 0) data[i] = 0;
        if (i % 13 == 0) data[i] = 0;
        if (i % 17 == 0) data[i] = 0;
        if (i % 19 == 0) data[i] = 0;
        if (i % 23 == 0) data[i] = 0;
        if (i % 29 == 0) data[i] = 0;
        if (i % 31 == 0) data[i] = 0;
        if (i % 37 == 0) data[i] = 0;
        if (i % 41 == 0) data[i] = 0;
        if (i % 43 == 0) data[i] = 0;
        if (i % 47 == 0) data[i] = 0;
        if (i % 51 == 0) data[i] = 0;
        if (i % 53 == 0) data[i] = 0;
    }

    // for (auto i = 0; i < 100; i++) {
    //     cout << data[i] << endl;
    // }

    auto d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, m * m);
    int  nnz1   = 0;
    int  nnz2   = 0;

    string fo1 = fi + ".csr1";
    string fo2 = fi + ".csr2";

    cusz::impl::GatherAsCSR(d_data, m * m, m, m, m, &nnz1, &fo1);
    cusz::impl::PruneGatherAsCSR(d_data, m * m, m, m, m, nnz2, &fo2);

    cout << "nnz1: " << nnz1 << endl;
    cout << "nnz2: " << nnz2 << endl;

    auto d_outlier1 = mem::CreateCUDASpace<float>(m * m);
    cusz::impl::ScatterFromCSR(d_outlier1, m * m, m /*lda*/, m /*m*/, m /*n*/, &nnz1, &fo1);
    auto outlier1 = mem::CreateHostSpaceAndMemcpyFromDevice(d_outlier1, m * m);

    auto d_outlier2 = mem::CreateCUDASpace<float>(m * m);
    cusz::impl::ScatterFromCSR(d_outlier2, m * m, m /*lda*/, m /*m*/, m /*n*/, &nnz2, &fo2);
    auto outlier2 = mem::CreateHostSpaceAndMemcpyFromDevice(d_outlier2, m * m);

    for (auto i = 0; i < len; i++) {
        if (outlier1[i] != outlier2[i]) {
            cout << i << endl;
            cout << outlier1[i] << endl;
            cout << outlier2[i] << endl;
            cout << data[i] << endl;
            break;
        }
    }
}
