/**
 * @file base_compressor.cu
 * @author Jiannan Tian
 * @brief Predictor-only Base Compressor; can also be used for dryrun.
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "dryrun.hh"
#include "pipeline/dryrun.inl"

template class cusz::Dryrunner<DataTrait<4>::type>;
template class cusz::Dryrunner<DataTrait<8>::type>;
