/**
 * @file hfclass.hip
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

#include "busyheader.hh"
#include "hf/hf.hh"
#include "hf/hfbk.hh"
// #include "hf/hfbk.cu.hh"
#include "hf/hfcodec.hh"
#include "mem/memseg_cxx.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/format.hh"

// deps
#include "port.hh"

// definitions
#include "detail/hf_g.inl"

#define HUFFCOARSE_CC(E, ETF, H, M)                   \
  template class cusz::HuffmanCodec<                  \
      ErrCtrlTrait<E, ETF>::type, HuffTrait<H>::type, \
      MetadataTrait<M>::type>;

// 23-06-04 restricted to u4 for quantization code

// HUFFCOARSE_CC(1, false, 4, 4)  // uint
// HUFFCOARSE_CC(1, false, 8, 4)  //
// HUFFCOARSE_CC(2, false, 4, 4)  //
// HUFFCOARSE_CC(2, false, 8, 4)  //
HUFFCOARSE_CC(4, false, 4, 4)  //
HUFFCOARSE_CC(4, false, 8, 4)  //

// HUFFCOARSE_CC(4, true, 4, 4)  // float
// HUFFCOARSE_CC(4, true, 8, 4)  //

#undef HUFFCOARSE_CC
