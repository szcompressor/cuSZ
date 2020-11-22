/**
 * @file par_huffman_sortbyfreq.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief (header) Sorts quantization codes by frequency, using a key-value sort.
 *        This functionality is placed in a separate compilation unit
 *        as thrust calls fail in par_huffman.cu.
 * @version 0.1
 * @date 2020-09-21
 * Created on: 2020-07
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

// clang-format off
namespace lossless { namespace par_huffman { namespace helper {
template <typename K, typename V> void SortByFreq(K*, V*, int);
}}}
// clang-format on
