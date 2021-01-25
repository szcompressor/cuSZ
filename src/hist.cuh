/**
 * @file histogram.cuh
 * @author Cody Rivera (cjrivera1@crimson.ua.edu), Megan Hickman Fulp (mlhickm@g.clemson.edu)
 * @brief Fast histogramming from [Gómez-Luna et al. 2013] (header)
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-16
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef HIST_CUH
#define HIST_CUH

#include <cuda_runtime.h>
#include <cstdio>

namespace data_process {
namespace reduce {

__global__ void NaiveHistogram(int input_data[], int output[], int N, int symbols_per_thread);

const static unsigned int WARP_SIZE = 32;
#define MIN(a, b) ((a) < (b)) ? (a) : (b)

/* Copied from J. Gomez-Luna et al */
template <typename Input_UInt, typename Output_UInt>
__global__ void p2013Histogram(Input_UInt*, Output_UInt*, size_t, int, int);

template <typename Input_Int, typename Output_UInt>
__global__ void p2013Histogram_int_input(Input_Int*, Output_UInt*, size_t, int, int, int);

}  // namespace reduce
}  // namespace data_process

#endif
