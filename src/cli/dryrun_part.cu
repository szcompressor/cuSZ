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

#include "dryrun_part.cuh"

template class cusz::BaseCompressor<cusz::PredictionUnified<  //
    DataTrait<4>::type,
    ErrCtrlTrait<2>::type,
    FastLowPrecisionTrait<true>::type>>;
