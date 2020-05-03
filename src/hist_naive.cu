#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include "__cuda_error_handling.cu"
#include "__io.hh"
#include "histogram.cuh"
#include "huffman_host_device.hh"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "<program> <datum name> <len>" << endl;
        exit(1);
    }
    string fname(argv[1]);
    size_t len   = atoi(argv[2]);
    auto   bcode = io::ReadBinaryFile<uint16_t>(fname, len);

    uint16_t*     _d_bcode;
    unsigned int* _d_freq;
    int           numBuckets = 1024;

    cudaMalloc(&_d_bcode, len * sizeof(uint16_t));
    cudaMalloc(&_d_freq, 1024 * sizeof(unsigned int));
    cudaMemcpy(_d_bcode, bcode, len * sizeof(uint16_t), cudaMemcpyHostToDevice);

    size_t hist_n_thread_old = 1024, hist_sym_per_thread_old = 1;                             // histograming
    size_t hist_n_block_old = (len - 1) / (hist_n_thread_old * hist_sym_per_thread_old) + 1;  // |
    prototype::GPU_Histogram<<<hist_n_block_old, hist_n_thread_old>>>(_d_bcode, _d_freq, len, 1);

    auto _h_freq = new int[1024]();
    cudaMemcpy(_h_freq, _d_freq, 1024 * sizeof(int), cudaMemcpyDeviceToHost);

    size_t non_zero_freq = 0;
    for (size_t i = 0; i < numBuckets; i++) {
        auto freq = _h_freq[i];
        if (freq == 0) continue;
        non_zero_freq++;
    }
    //    cout << "non-zero-freq #: " << non_zero_freq << endl;
    //
    //    for (size_t i = numBuckets / 2 - 10; i < numBuckets / 2 + 10; i++) {
    //        auto freq = _h_freq[i];
    //        if (freq == 0) continue;
    //        printf("%5lu: %-d\n", i, *(_h_freq + i));
    //    }
    for (size_t i = 0; i < 1024; i++) {
        if (_h_freq[i] == 0) continue;
        cout << i << ": " << _h_freq[i] << "\t" << endl;
    }

    string hist_name(fname + ".hist_naive");
    io::WriteBinaryFile(_h_freq, numBuckets, &hist_name);

    auto sum = accumulate(_h_freq, _h_freq + 1024, (size_t)0);
    cout << "sum: " << sum << endl;

    return 0;
}
