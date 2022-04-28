/**
 * @file compressor.cu
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2021-10-05
 * (create) 2020-02-12; (release) 2020-09-20;
 * (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06; (rev.4) 2021-10-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "detail/compressor_impl.cuh"
#include "framework.hh"

template class cusz::Compressor<cusz::PredefinedCombination<float>::LorenzoFeatured>::impl;
template class cusz::Compressor<cusz::PredefinedCombination<float>::Spline3Featured>::impl;  // TODO
