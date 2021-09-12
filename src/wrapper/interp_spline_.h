/**
 * @file interp_spline.h
 * @author Jiannan Tian
 * @brief (header) A high-level Spline3D wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_INTERP_SPLINE_H
#define CUSZ_WRAPPER_INTERP_SPLINE_H

template <typename DataIter = float*, typename QuantIter = unsigned short*, typename FP = float>
class Spline3 {
    unsigned int dimx, dimy, dimz;
    unsigned int nblockx, nblocky, nblockz;

    FP eb_r, ebx2, ebx2_r;

    static const auto BLOCK = 137;

   public:
    Spline3() = default;

    int reversed_predict_quantize() { return 0; }
    int predict_quantize() { return 0; }
    int get_len_quant() { return 0; }
    int get_len_helper() { return 0; }
};

#endif