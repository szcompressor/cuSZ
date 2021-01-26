/**
 * @file lorenzo_trait.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2020-09-23
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef LORENZO_TRAIT_CUH
#define LORENZO_TRAIT_CUH

#include "cuda_wrap.cuh"
#include "dryrun.cuh"
#include "dualquant.cuh"
#include "metadata.hh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

namespace kernel_v2 = cusz::predictor_quantizer::v2;
namespace kernel_v3 = cusz::predictor_quantizer::v3;

enum class workflow { zip, unzip, fm_unzip };

template <int ndim, typename Data, workflow w>
struct LorenzoNdConfig {
    static const int B = MetadataTrait<ndim>::Block;
    kernel_cfg_t     cfg;
    lorenzo_zip      z_ctx;
    lorenzo_unzip    x_ctx;
    lorenzo_dryrun   r_ctx;

    LorenzoNdConfig(Integer4 dims, Integer4 strides, Integer4 nblks, int radius, double eb)
    {
        z_ctx.d0 = dims._0, z_ctx.d1 = dims._1, z_ctx.d2 = dims._2;
        z_ctx.stride1 = strides._1, z_ctx.stride2 = strides._2;
        z_ctx.radius = radius;
        z_ctx.ebx2_r = 1 / (2 * eb);

        r_ctx.d0 = dims._0, r_ctx.d1 = dims._1, r_ctx.d2 = dims._2;
        r_ctx.stride1 = strides._1, r_ctx.stride2 = strides._2;
        r_ctx.radius = radius;
        r_ctx.ebx2_r = 1 / (2 * eb);
        r_ctx.ebx2   = 2 * eb;

        x_ctx.d0 = dims._0, x_ctx.d1 = dims._1, x_ctx.d2 = dims._2;
        x_ctx.stride1 = strides._1, x_ctx.stride2 = strides._2;
        x_ctx.nblk0 = nblks._0, x_ctx.nblk1 = nblks._1, x_ctx.nblk2 = nblks._2;
        x_ctx.radius = radius;
        x_ctx.ebx2   = 2 * eb;

        if CONSTEXPR (w == workflow::zip or w == workflow::fm_unzip) {
            if CONSTEXPR (ndim == 1) {
                cfg.Dg = dim3(nblks._0);
                cfg.Db = dim3(B);
                cfg.Ns = B * sizeof(Data);
            }
            else if CONSTEXPR (ndim == 2) {
                cfg.Dg = dim3(nblks._0, nblks._1);
                cfg.Db = dim3(B, B);
                cfg.Ns = B * B * sizeof(Data);
            }
            else if CONSTEXPR (ndim == 3) {
                cfg.Dg = dim3(nblks._0, nblks._1, nblks._2);
                cfg.Db = dim3(B, B, B);
                cfg.Ns = B * B * B * sizeof(Data);
            }
            cfg.S = nullptr;
        }
        else if CONSTEXPR (w == workflow::unzip) {  // unzip
            if CONSTEXPR (ndim == 1) {
                cfg.Dg = dim3((nblks._0 + B - 1) / B);
                cfg.Db = dim3(B);
            }
            else if CONSTEXPR (ndim == 2) {
                cfg.Dg = dim3((nblks._0 + B - 1) / B, (nblks._1 + B - 1) / B);
                cfg.Db = dim3(B, B);
            }
            else if CONSTEXPR (ndim == 3) {
                cfg.Dg = dim3((nblks._0 + B - 1) / B, (nblks._1 + B - 1) / B, (nblks._2 + B - 1) / B);
                cfg.Db = dim3(B, B, B);
            }
            cfg.Ns = 0;
            cfg.S  = nullptr;
        }
    }
};

/////////

// clang-format off
namespace zip    { template <int ndim> struct Lorenzo_nd1l; }
namespace dryrun { template <int ndim> struct Lorenzo_nd1l; }
namespace unzip  { template <int ndim> struct Lorenzo_nd1l; }
// clang-format on

template <int ndim>
struct zip::Lorenzo_nd1l {
    template <typename Data, typename Quant>
    static void Call(lorenzo_zip ctx, Data* d, Quant* q)
    {
        if CONSTEXPR (ndim == 1) kernel_v2::c_lorenzo_1d1l<Data, Quant>(ctx, d, q);
        if CONSTEXPR (ndim == 2) kernel_v3::c_lorenzo_2d1l<Data, Quant>(ctx, d, q);
        if CONSTEXPR (ndim == 3) kernel_v3::c_lorenzo_3d1l<Data, Quant>(ctx, d, q);
    }
};

template <int ndim>
struct unzip::Lorenzo_nd1l {
    template <typename Data, typename Quant>
    static void Call(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
    {
        if CONSTEXPR (ndim == 1) kernel_v2::x_lorenzo_1d1l<Data, Quant>(ctx, xd, outlier, q);
        if CONSTEXPR (ndim == 2) kernel_v3::x_lorenzo_2d1l<Data, Quant>(ctx, xd, outlier, q);
        if CONSTEXPR (ndim == 3) kernel_v3::x_lorenzo_3d1l<Data, Quant>(ctx, xd, outlier, q);
    }
};

// template <int ndim>
// struct dryrun::Lorenzo_nd1l {
//    template <typename Data>
//    static void Call(struct Metadata* m, Data* d)
//    {
//        if CONSTEXPR (ndim == 1) cusz::dryrun::lorenzo_1d1l(m, d);
//        if CONSTEXPR (ndim == 2) cusz::dryrun::lorenzo_2d1l(m, d);
//        if CONSTEXPR (ndim == 3) cusz::dryrun::lorenzo_3d1l(m, d);
//    }
//};

#endif
