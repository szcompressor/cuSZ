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
#include "metadata.hh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

enum class workflow { zip, unzip };

template <int ndim, typename Data, workflow w>
struct LorenzoNdConfig {
    static const int B = MetadataTrait<ndim>::Block;
    kernel_cfg_t     cfg;
    lorenzo_zip      z_ctx;
    lorenzo_unzip    x_ctx;
    lorenzo_dryrun   r_ctx;

    LorenzoNdConfig(UInteger4 dims, UInteger4 strides, UInteger4 nblks, int radius, double eb)
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

        if CONSTEXPR (w == workflow::zip or w == workflow::unzip) {
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
    }
};

#endif
