/**
 * @file codec.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2023-06-02
 * (created) 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National Laboratory
 * @copyright (C) 2023 by Indiana University
 *
 */

#include "common/type_traits.hh"

#include "hf/hf.hh"
#include "hf/hf_bookg.hh"
#include "hf/hf_codecg.hh"

#include "detail/hf_g.inl"

#define HUFFCOARSE_CC(E, ETF, H, M) \
    template class cusz::HuffmanCodec<ErrCtrlTrait<E, ETF>::type, HuffTrait<H>::type, MetadataTrait<M>::type>;

HUFFCOARSE_CC(1, false, 4, 4)  // uint
HUFFCOARSE_CC(1, false, 8, 4)  //
HUFFCOARSE_CC(2, false, 4, 4)  //
HUFFCOARSE_CC(2, false, 8, 4)  //
HUFFCOARSE_CC(4, false, 4, 4)  //
HUFFCOARSE_CC(4, false, 8, 4)  //

// HUFFCOARSE_CC(4, true, 4, 4)  // float
// HUFFCOARSE_CC(4, true, 8, 4)  //

#undef HUFFCOARSE_CC
