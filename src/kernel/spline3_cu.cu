/**
 * @file spline3.cu
 * @author Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "detail/spline3.inl"
#include "kernel/spline.hh"
#include "busyheader.hh"
#include "cusz/type.h"

void spline3_calc_sizes(void* _l3, Spline3Len3* s3l3)
{
    constexpr auto BLOCK = 8;

    auto div       = [](auto _len, auto _sublen) { return (_len - 1) / _sublen + 1; };
    auto linearize = [](dim3 a) { return a.x * a.y * a.z; };
    auto leap      = [](dim3 a) { return dim3(1, a.x, a.x * a.y); };

    auto l3 = *((dim3*)_l3);

    // original
    s3l3->len          = l3.x * l3.y * l3.z;
    s3l3->l3_data      = l3;
    s3l3->stride3_data = dim3(1, l3.x, l3.x * l3.y);

    // padding
    s3l3->grid_dim.x      = div(l3.x, BLOCK * 4);
    s3l3->grid_dim.y      = div(l3.y, BLOCK);
    s3l3->grid_dim.z      = div(l3.z, BLOCK);
    s3l3->x_32            = s3l3->grid_dim.x * 32;  // e.g., 235 -> 256
    s3l3->y_8             = s3l3->grid_dim.y * 8;   // e.g., 449 -> 456
    s3l3->z_8             = s3l3->grid_dim.z * 8;   // e.g., 449 -> 456
    s3l3->l3_aligned      = dim3(s3l3->x_32, s3l3->y_8, s3l3->z_8);
    s3l3->len_aligned     = linearize(s3l3->l3_aligned);
    s3l3->stride3_aligned = leap(s3l3->l3_aligned);

    // anchor point
    s3l3->l3_anchor.x    = div(l3.x, BLOCK);
    s3l3->l3_anchor.y    = div(l3.y, BLOCK);
    s3l3->l3_anchor.z    = div(l3.z, BLOCK);
    s3l3->len_anchor     = linearize(s3l3->l3_anchor);
    s3l3->stride3_anchor = leap(s3l3->l3_anchor);

    printf("\ncalculating spline3 sizes:\n");
    printf("data-(x, y, z)=(%u, %u, %u)\n", s3l3->l3_data.x, s3l3->l3_data.y, s3l3->l3_data.z);
    printf("stride_data-(x, y, z)=(%u, %u, %u)\n", s3l3->stride3_data.x, s3l3->stride3_data.y, s3l3->stride3_data.z);
    printf("padded-(x, y, z)=(%u, %u, %u)\n", s3l3->x_32, s3l3->y_8, s3l3->z_8);
    printf(
        "stride_aligned-(x, y, z)=(%u, %u, %u)\n", s3l3->stride3_aligned.x, s3l3->stride3_aligned.y,
        s3l3->stride3_aligned.z);
    printf("anchor-(x, y, z)=(%u, %u, %u)\n", s3l3->l3_anchor.x, s3l3->l3_anchor.y, s3l3->l3_anchor.z);
    printf(
        "spline3 quant-code size: %u (%.3lfx the original)\n", s3l3->len_aligned, s3l3->len_aligned * 1.0 / s3l3->len);
    printf("\n");
}

template <typename T, typename E, typename FP>
int spline_construct(
    pszmem_cxx<T>* data,
    pszmem_cxx<T>* anchor,
    pszmem_cxx<E>* ectrl,
    double         eb,
    uint32_t       radius,
    void*          stream)
{
    constexpr auto BLOCK = 8;

    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    auto l3       = data->template len3<dim3>();
    auto grid_dim = dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

    cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256, false>  //
        <<<grid_dim, dim3(256, 1, 1), 0, (cudaStream_t)stream>>>(
            data->dptr(), data->template len3<dim3>(), data->template st3<dim3>(),     //
            ectrl->dptr(), ectrl->template len3<dim3>(), ectrl->template st3<dim3>(),  //
            anchor->dptr(), anchor->template st3<dim3>(), eb_r, ebx2, radius);

    cudaStreamSynchronize((cudaStream_t)stream);

    return 0;
}

template <typename T, typename E, typename FP>
int spline_reconstruct(
    pszmem_cxx<T>* anchor,
    pszmem_cxx<E>* ectrl,
    pszmem_cxx<T>* xdata,
    double         eb,
    uint32_t       radius,
    void*          stream)
{
    constexpr auto BLOCK = 8;

    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    auto l3       = xdata->template len3<dim3>();
    auto grid_dim = dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

    cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256>                          //
        <<<grid_dim, dim3(256, 1, 1), 0, (cudaStream_t)stream>>>                       //
        (ectrl->dptr(), ectrl->template len3<dim3>(), ectrl->template st3<dim3>(),     //
         anchor->dptr(), anchor->template len3<dim3>(), anchor->template st3<dim3>(),  //
         xdata->dptr(), xdata->template len3<dim3>(), xdata->template st3<dim3>(),     //
         eb_r, ebx2, radius);

    cudaStreamSynchronize((cudaStream_t)stream);
    return 0;
}

#define INIT(T, E)                                                                                        \
    template int spline_construct<T, E>(                                                                  \
        pszmem_cxx<T> * data, pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl, double eb, uint32_t radius,  \
        void* stream);                                                                                    \
    template int spline_reconstruct<T, E>(                                                                \
        pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl, pszmem_cxx<T> * xdata, double eb, uint32_t radius, \
        void* stream);

INIT(f4, u1)
INIT(f4, u2)
INIT(f4, u4)
INIT(f4, f4)

INIT(f8, u1)
INIT(f8, u2)
INIT(f8, u4)
INIT(f8, f4)

#undef INIT
