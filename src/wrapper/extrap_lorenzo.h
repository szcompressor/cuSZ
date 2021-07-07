/**
 * @file extrap_lorenzo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_EXTRAP_LORENZO_H
#define CUSZ_WRAPPER_EXTRAP_LORENZO_H

template <typename Data = float, typename Quant = float, typename FP = float>
void tbd_lorenzo_dryrun(Data* data, dim3 size3, int ndim, FP eb);

template <typename Data = float, typename Quant = float, typename FP = float, bool DELAY_POSTQUANT = false>
void compress_lorenzo_construct(Data* data, Quant* quant, dim3 size3, int ndim, FP eb, int radius, float& ms);

template <typename Data = float, typename Quant = float, typename FP = float, bool DELAY_POSTQUANT = false>
void decompress_lorenzo_reconstruct(Data* data, Quant* quant, dim3 size3, int ndim, FP eb, int radius, float& ms);

#endif