/**
 * @file par_huffman_sortbyfreq.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Sorts quantization codes by frequency, using a key-value sort.
 *        This functionality is placed in a separate compilation unit
 *        as thrust calls fail in par_huffman.cu.
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2020-07
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdint>

template <typename K, typename V>
void SortByFreq(K* freq, V* qcode, int size)
{
    using namespace thrust;
    sort_by_key(
        device_ptr<K>(freq),         //
        device_ptr<K>(freq + size),  //
        device_ptr<V>(qcode));
}

template void SortByFreq<unsigned int, uint8_t>(unsigned int*, uint8_t*, int);
template void SortByFreq<unsigned int, uint16_t>(unsigned int*, uint16_t*, int);
template void SortByFreq<unsigned int, uint32_t>(unsigned int*, uint32_t*, int);
