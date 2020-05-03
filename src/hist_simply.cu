#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include "__cuda_error_handling.cu"
#include "__io.hh"
#include "histogram.cuh"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "<program> <datum name> <len>" << endl;
        exit(1);
    }
    cout << "[INFO] Currently, we use uint16_t to represent bincode" << endl;
    string fname(argv[1]);
    size_t len = atoi(argv[2]);

    int maxbytes = 98304;

    auto bcode = io::ReadBinaryFile<uint16_t>(fname, len);
    cudaFuncSetAttribute(p2013Histogram<uint16_t, unsigned int>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    uint16_t* _d_bcode;
    cudaMalloc(&_d_bcode, len * sizeof(uint16_t));

    unsigned int* _d_freq;
    int           numBuckets = 1024;
    int           numValues  = len;

    cudaMalloc(&_d_freq, (unsigned long)1024 * sizeof(unsigned int));
    cudaMemcpy(_d_bcode, bcode, len * sizeof(uint16_t), cudaMemcpyHostToDevice);

    /* Sets up parameters */
    int numSMs         = 84;
    int itemsPerThread = 1;

    int RPerBlock = (maxbytes / sizeof(int)) / (numBuckets + 1);
    int numBlocks = numSMs;

    /* Fits to size */
    int threadsPerBlock = ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
    while (threadsPerBlock > 1024) {
        if (RPerBlock <= 1) {
            threadsPerBlock = 1024;
        } else {
            RPerBlock /= 2;
            numBlocks *= 2;
            threadsPerBlock = ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
        }
    }

    // HANDLE_ERROR(cudaEventRecord(start));
    cudaMemset(_d_freq, 0, sizeof(int) * numBuckets);
    p2013Histogram<<<numBlocks, threadsPerBlock, ((numBuckets + 1) * RPerBlock) * sizeof(int)>>>(_d_bcode, _d_freq, numValues, numBuckets, RPerBlock);
    HANDLE_ERROR(cudaDeviceSynchronize());

    auto _h_freq = new int[1024]();
    cudaMemcpy(_h_freq, _d_freq, 1024 * sizeof(int), cudaMemcpyDeviceToHost);

    size_t non_zero_freq = 0;
    for (size_t i = 0; i < numBuckets; i++) {
        auto freq = _h_freq[i];
        if (freq == 0) continue;
        non_zero_freq++;
    }
    for (size_t i = 0; i < 1024; i++) {
        if (_h_freq[i] == 0) continue;
        cout << i << ": " << _h_freq[i] << "\t" << endl;
    }

    string hist_name(fname + ".hist");
    io::WriteBinaryFile(_h_freq, numBuckets, &hist_name);

    auto sum = accumulate(_h_freq, _h_freq + 1024, (size_t)0);
    cout << "sum: " << sum << endl;

    return 0;
}
