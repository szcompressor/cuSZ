/**
 * @file hf_obj_cu.cu
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

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "busyheader.hh"
#include "hf/hf.hh"
#include "hf/hfbk.hh"
#include "hf/hfcodec.hh"
#include "mem/cxx_memobj.h"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/format.hh"

// definitions
#include "detail/hfclass.dp.inl"

template class phf::HuffmanCodec<u1, u4>;
template class phf::HuffmanCodec<u2, u4>;
template class phf::HuffmanCodec<u4, u4>;
