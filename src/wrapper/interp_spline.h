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

#ifndef CUSZ_INTERP_WRAPPER_SPLINE_H
#define CUSZ_INTERP_WRAPPER_SPLINE_H

// malloc at function call
void spline3_configure(dim3 in_size3, size_t* quantcode_len, size_t* anchor_len);

// accept device pointer, whose size is known
// internal data partition is fixed
template <typename DataIter = float*, typename QuantIter = float*, typename FP = float>
void compress_spline3(DataIter in, dim3 in_size3, QuantIter out, dim3 out_size, FP eb);

template <typename DataIter = float*, typename QuantIter = float*>
void decompress_spline3(dim3 size);

#endif