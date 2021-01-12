#ifndef DATASETS_HH
#define DATASETS_HH
/**
 * @file datasets.hh
 * @author Jiannan Tian
 * @brief Demonstrative datasets with prefilled dimensions from https://sdrbench.github.io (header)
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-11
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <string>
#include <unordered_map>

#include "constants.hh"
#include "types.hh"

size_t* InitializeDemoDims(
    std::string const& datum,
    size_t             cap,
    bool               override = false,
    size_t             new_d0   = 1,
    size_t             new_d1   = 1,
    size_t             new_d2   = 1,
    size_t             new_d3   = 1);

#endif
