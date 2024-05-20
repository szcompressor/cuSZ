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
#include "typing.hh"

// deps
#include <cuda.h>
#include "port.hh"
// definitions
#include "detail/hfclass.cu_hip.inl"

template class cusz::HuffmanCodec<u1, u4, true>;
template class cusz::HuffmanCodec<u2, u4, true>;
template class cusz::HuffmanCodec<u4, u4, true>;

template class cusz::HuffmanCodec<u1, u4, false>;
template class cusz::HuffmanCodec<u2, u4, false>;
template class cusz::HuffmanCodec<u4, u4, false>;
