// Cody Rivera <cjrivera1@crimson.ua.edu>
// Megan Hickman

#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include <cuda_runtime.h>
#include <cstdio>

__global__ void naiveHistogram(int input_data[], int output[], int N, int symbols_per_thread);

const static unsigned int WARP_SIZE = 32;
#define MIN(a, b) ((a) < (b)) ? (a) : (b)

// Optimized 2013
/* Copied from J. Gomez-Luna et al */
template <typename T, typename Q>
__global__ void p2013Histogram(T* input_data, Q* output, size_t N, int bins, int R);

#endif
