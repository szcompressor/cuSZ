/**
 * @file hfclass.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2023-06-02
 * (created) 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National
 * Laboratory
 * @copyright (C) 2023 by Indiana University
 *
 */

#include "hfclass.hh"

// deps
#include <cuda.h>
// definitions
#include "detail/hfclass.cuhip.inl"

template class phf::HuffmanCodec<u1>;
template class phf::HuffmanCodec<u2>;
template class phf::HuffmanCodec<u4>;
