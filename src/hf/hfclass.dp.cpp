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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "busyheader.hh"
#include "hf/hf.hh"
#include "hf/hfbk.hh"
#include "hf/hfcodec.hh"
#include "mem/memseg_cxx.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/format.hh"

// deps
#include "port.hh"
// definitions
#include "detail/hfclass.dp.inl"

template class cusz::HuffmanCodec<u1, u4>;
template class cusz::HuffmanCodec<u2, u4>;
template class cusz::HuffmanCodec<u4, u4>;

