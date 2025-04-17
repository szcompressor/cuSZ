/**
 * Originally from par_huffman_sortbyfreq.cu by Cody Rivera (cjrivera1@crimson.ua.edu)
 * Sorts quantization codes by frequency, using a key-value sort. This functionality is placed in a separate
 * compilation unit as thrust calls fail in par_huffman.cu.
 *
 * Resolved by using `thrust::device_pointer_cast(var)` instead of `thrust::device_pointer<T>(var)`
 */

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

void lambda_sort_by_freq(uint32_t* freq, const int len, uint32_t* qcode)
{
    thrust::sort_by_key(
        thrust::device_pointer_cast(freq), thrust::device_pointer_cast(freq + len), thrust::device_pointer_cast(qcode));
};