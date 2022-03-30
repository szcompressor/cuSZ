// modified from thrust example
// attach the license below when push to master branch
// https://github.com/NVIDIA/thrust/blob/main/LICENSE

/**
 * @file rle.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-01
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef KERNEL_RLE_CUH
#define KERNEL_RLE_CUH

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <iostream>
#include <iterator>

using const_gen = thrust::constant_iterator<int>;
using counter   = thrust::counting_iterator<int>;

namespace kernel {

template <typename T>
void RunLengthEncoding(T* d_fullfmt_data, const size_t N, T* d_compact_data, int* d_lengths, size_t& num_runs)
{
    thrust::device_ptr<T>   input   = thrust::device_pointer_cast(d_fullfmt_data);
    thrust::device_ptr<T>   output  = thrust::device_pointer_cast(d_compact_data);
    thrust::device_ptr<int> lengths = thrust::device_pointer_cast(d_lengths);
    // compute the output size (run lengths)
    num_runs = thrust::reduce_by_key(
                   input, input + N,  // input::key (symbol)
                   const_gen(1),      // input::value (count)
                   output,            // output::key (symbol)
                   lengths)           // output::value (count)
                   .first -
               output;
}

template <typename T>
void RunLengthDecoding(T* d_fullfmt_data, const size_t N, T* d_compact_data, int* d_lengths, const size_t num_runs)
{
    thrust::device_ptr<T>   output  = thrust::device_pointer_cast(d_fullfmt_data);
    thrust::device_ptr<T>   input   = thrust::device_pointer_cast(d_compact_data);
    thrust::device_ptr<int> lengths = thrust::device_pointer_cast(d_lengths);

    // scan the lengths
    thrust::inclusive_scan(lengths, lengths + num_runs, lengths);

    // compute input index for each output element
    thrust::device_vector<int> indices(N);
    thrust::lower_bound(
        lengths, lengths + N,        //
        counter(1), counter(N + 1),  //
        indices.begin());

    thrust::encode(indices.begin(), indices.end(), input, output);
}

}  // namespace kernel

#endif
