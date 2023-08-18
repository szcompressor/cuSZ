/**
 * @file _canonical.cuh
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef B684F0FA_8869_4DDF_9467_2E28E967AC06
#define B684F0FA_8869_4DDF_9467_2E28E967AC06

#include <cstdint>
#include <cstring>

#include "cusz/type.h"

template <typename E, typename H>
class hf_space {
 public:
  static const int TYPE_BITS = sizeof(H) * 8;

  static u4 space_bytes(int const bklen)
  {
    return sizeof(H) * (3 * bklen) + sizeof(u4) * (4 * TYPE_BITS) +
           sizeof(E) * bklen;
  }

  static u4 revbook_bytes(int const bklen)
  {
    return sizeof(u4) * (2 * TYPE_BITS) + sizeof(E) * bklen;
  }

  static u4 revbook_offset(int const bklen)
  {
    return sizeof(H) * (3 * bklen) + sizeof(u4) * (2 * TYPE_BITS);
  }
};

template <typename E, typename H>
int canonize_on_gpu(uint8_t* bin, uint32_t bklen, void* stream);

template <typename E, typename H>
int canonize(uint8_t* bin, uint32_t const bklen);

#endif /* B684F0FA_8869_4DDF_9467_2E28E967AC06 */
