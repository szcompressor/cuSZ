/**
 * @file ex_common.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-11-30
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef EX_COMMON_CUH
#define EX_COMMON_CUH

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include "../src/sp_path.cuh"

const static auto DEVICE      = cusz::LOC::DEVICE;
const static auto HOST        = cusz::LOC::HOST;
const static auto HOST_DEVICE = cusz::LOC::HOST_DEVICE;

using T = float;

void exp_sppath_lorenzo_quality(unsigned int len, std::string fname, double eb)
{
    Capsule<T> data(len);
    Capsule<T> xdata(len);

    data.alloc<HOST_DEVICE>().from_fs_to<HOST>(fname).host2device();
    xdata.alloc<HOST>();

    memcpy(xdata.template get<HOST>(), data.template get<HOST>(), sizeof(T) * len);

    // TODO failed to check if on device
    auto rng = data.prescan().get_rng();

    auto r2r_eb = eb * rng;
    auto ebx2   = r2r_eb * 2;
    auto ebx2_r = 1 / ebx2;

    cout << "r2r eb: " << eb * rng << endl;

    std::for_each(
        xdata.template get<HOST>(),                     //
        xdata.template get<HOST>() + len,               //
        [&](T& el) { el = round(el * ebx2_r) * ebx2; }  //
    );

    stat_t stat;
    analysis::verify_data<T>(&stat, xdata.template get<HOST>(), data.template get<HOST>(), len);
    analysis::print_data_quality_metrics<T>(&stat, 0, false);
}

#endif