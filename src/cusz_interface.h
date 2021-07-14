#ifndef CUSZ_INTERFACE_H
#define CUSZ_INTERFACE_H

/**
 * @file cusz_interface.h
 * @author Jiannan Tian
 * @brief Workflow of cuSZ (header).
 * @version 0.3
 * @date 2021-07-12
 * (created) 2020-02-12 (rev.1) 2020-09-20 (rev.2) 2021-07-12
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include "argparse.hh"
#include "datapack.hh"
#include "pack.hh"
#include "type_trait.hh"

using namespace std;

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(
    argpack*,
    struct PartialData<typename DataTrait<If_FP, DataByte>::Data>*,
    dim3,
    metadata_pack* mp,
    unsigned int = 1);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack*, metadata_pack* mp);

#endif
