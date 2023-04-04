/**
 * @file test_l2_cudaproto.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-02-25
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda/std/functional>
#include <iostream>
#include <nvfunctional>
#include <string>
#include <typeinfo>

#include "common/capsule.hh"
#include "kernel/detail/lorenzo_proto.inl"

using std::cout;
using std::endl;
namespace proto = psz::cuda::__kernel::proto;

using T  = float;
using FP = float;
using EQ = int32_t;

size_t t1d_len = 256;
dim3   t1d_len3{256, 1, 1};
dim3   t1d_stride3{1, 1, 1};
dim3   t1d_grid_dim{1, 1, 1};
dim3   t1d_block_dim{256, 1, 1};

size_t t2d_len = 256;
dim3   t2d_len3{16, 16, 1};
dim3   t2d_stride3{1, 16, 1};
dim3   t2d_grid_dim{1, 1, 1};
dim3   t2d_block_dim{16, 16, 1};

size_t t3d_len = 512;
dim3   t3d_len3{8, 8, 8};
dim3   t3d_stride3{1, 8, 64};
dim3   t3d_grid_dim{1, 1, 1};
dim3   t3d_block_dim{8, 8, 8};

#include "misc/correctness.inl"

bool test1(
    int         dim,
    T const*    h_input,
    size_t      len,
    dim3        len3,
    dim3        stride3,
    T const*    h_expected_output,
    std::string funcname)
{
    Capsule<T>  input(len);
    Capsule<T>  outlier(len);
    Capsule<EQ> eq(len);

    input.malloc().set_hptr(const_cast<T*>(h_input)).h2d();
    eq.malloc().mallochost();
    outlier.malloc();

    auto radius = 512;

    if (dim == 1)
        proto::c_lorenzo_1d1l<T>
            <<<t1d_grid_dim, t1d_block_dim>>>(input.dptr(), len3, stride3, radius, 1, eq.dptr(), outlier.dptr());
    else if (dim == 2)
        proto::c_lorenzo_2d1l<T>
            <<<t2d_grid_dim, t2d_block_dim>>>(input.dptr(), len3, stride3, radius, 1, eq.dptr(), outlier.dptr());
    else if (dim == 3)
        proto::c_lorenzo_3d1l<T>
            <<<t3d_grid_dim, t3d_block_dim>>>(input.dptr(), len3, stride3, radius, 1, eq.dptr(), outlier.dptr());
    cudaDeviceSynchronize();

    eq.d2h();

    // for (auto i = 0; i < len; i++) { cout << eq.hptr(i) << endl; }

    bool ok = true;
    for (auto i = 0; i < len; i++) {
        // subject to change according to the algorithm
        if (eq.hptr(i) - radius != h_expected_output[i]) {
            ok = false;
            break;
        }
    }
    cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

    input.free();
    eq.free().freehost();
    outlier.free();

    return ok;
}

bool test2(
    int         dim,
    EQ const*   _h_input,
    size_t      len,
    dim3        len3,
    dim3        stride3,
    T const*    h_expected_output,
    std::string funcname)
{
    auto radius = 512;

    Capsule<EQ> input(len);
    input.malloc().mallochost();

    for (auto i = 0; i < len; i++) input.hptr(i) = _h_input[i] + radius;
    input.h2d();

    Capsule<T> xdata(len);
    xdata.malloc().mallochost();

    if (dim == 1)
        proto::x_lorenzo_1d1l<T><<<t1d_grid_dim, t1d_block_dim>>>(
            input.dptr(), xdata.dptr() /* outlier */, len3, stride3, radius, 1, xdata.dptr());
    else if (dim == 2)
        proto::x_lorenzo_2d1l<T><<<t2d_grid_dim, t2d_block_dim>>>(
            input.dptr(), xdata.dptr() /* outlier */, len3, stride3, radius, 1, xdata.dptr());
    else if (dim == 3)
        proto::x_lorenzo_3d1l<T><<<t3d_grid_dim, t3d_block_dim>>>(
            input.dptr(), xdata.dptr() /* outlier */, len3, stride3, radius, 1, xdata.dptr());
    else {
        throw std::runtime_error("must be 1, 2, or 3D.");
    }

    cudaDeviceSynchronize();

    xdata.d2h();

    bool ok = true;
    for (auto i = 0; i < len; i++) {
        if (xdata.hptr(i) != h_expected_output[i]) {
            ok = false;
            break;
        }
    }
    cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

    input.free().freehost();
    xdata.free().freehost();

    return ok;
}

bool test3(int dim, T const* h_input, size_t len, dim3 len3, dim3 stride3, std::string funcname)
{
    Capsule<T> input(len);
    input.malloc().mallochost();
    for (auto i = 0; i < len; i++) input.hptr(i) = h_input[i];
    input.h2d();

    Capsule<EQ> eq(len);
    eq.malloc();

    Capsule<T> outlier(len);
    outlier.malloc();

    Capsule<T> xdata(len);
    xdata.malloc().mallochost();

    auto radius = 512;

    auto eb     = 1e-2;
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / (eb * 2);

    if (dim == 1) {
        proto::c_lorenzo_1d1l<T>
            <<<t1d_grid_dim, t1d_block_dim>>>(input.dptr(), len3, stride3, radius, ebx2_r, eq.dptr(), outlier.dptr());
        cudaDeviceSynchronize();
        proto::x_lorenzo_1d1l<T><<<t1d_grid_dim, t1d_block_dim>>>(
            eq.dptr(), xdata.dptr() /* outlier */, len3, stride3, radius, ebx2, xdata.dptr());
        cudaDeviceSynchronize();
    }
    else if (dim == 2) {
        proto::c_lorenzo_2d1l<T>
            <<<t2d_grid_dim, t2d_block_dim>>>(input.dptr(), len3, stride3, radius, ebx2_r, eq.dptr(), outlier.dptr());
        cudaDeviceSynchronize();
        proto::x_lorenzo_2d1l<T><<<t2d_grid_dim, t2d_block_dim>>>(
            eq.dptr(), xdata.dptr() /* outlier */, len3, stride3, radius, ebx2, xdata.dptr());
        cudaDeviceSynchronize();
    }
    else if (dim == 3) {
        proto::c_lorenzo_3d1l<T>
            <<<t3d_grid_dim, t3d_block_dim>>>(input.dptr(), len3, stride3, radius, ebx2_r, eq.dptr(), outlier.dptr());
        cudaDeviceSynchronize();
        proto::x_lorenzo_3d1l<T><<<t3d_grid_dim, t3d_block_dim>>>(
            eq.dptr(), xdata.dptr() /* outlier */, len3, stride3, radius, ebx2, xdata.dptr());
        cudaDeviceSynchronize();
    }

    // cudaMemcpy(h_xdata, xdata, sizeof(EQ) * len, cudaMemcpyDeviceToHost);
    xdata.d2h();

    // for (auto i = 0; i < len; i++) { cout << h_xdata[i] << endl; }

    bool ok = true;
    for (auto i = 0; i < len; i++) {
        if (xdata.hptr(i) != h_input[i]) {
            ok = false;
            break;
        }
    }
    cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

    // input.debug();

    input.free().freehost();
    eq.free();
    outlier.free();
    xdata.free().freehost();

    return ok;
}

bool run_test1()
{
    auto all_pass = true;

    all_pass = all_pass and test1(1, t1d_in, t1d_len, t1d_len3, t1d_stride3, t1d_comp_out, "standalone cl1d1l");
    all_pass = all_pass and test1(2, t2d_in, t2d_len, t2d_len3, t2d_stride3, t2d_comp_out, "standalone cl2d1l");
    all_pass = all_pass and test1(3, t3d_in, t3d_len, t3d_len3, t3d_stride3, t3d_comp_out, "standalone cl3d1l");

    return all_pass;
}

bool run_test2()
{
    auto all_pass = true;

    all_pass = all_pass and test2(1, t1d_eq, t1d_len, t1d_len3, t1d_stride3, t1d_decomp_out, "standalone xl1d1l");
    all_pass = all_pass and test2(2, t2d_eq, t2d_len, t2d_len3, t2d_stride3, t2d_decomp_out, "standalone xl2d1l");
    all_pass = all_pass and test2(3, t3d_eq, t3d_len, t3d_len3, t3d_stride3, t3d_decomp_out, "standalone xl3d1l");

    return all_pass;
}

bool run_test3()
{
    auto all_pass = true;

    all_pass = all_pass and test3(1, t1d_in, t1d_len, t1d_len3, t1d_stride3, "lorenzo_1d1l");
    all_pass = all_pass and test3(2, t2d_in, t2d_len, t2d_len3, t2d_stride3, "lorenzo_2d1l");
    all_pass = all_pass and test3(3, t3d_in, t3d_len, t3d_len3, t3d_stride3, "lorenzo_3d1l");

    return all_pass;
}

int main()
{
    auto all_pass = true;

    all_pass = all_pass and run_test1();
    all_pass = all_pass and run_test2();
    all_pass = all_pass and run_test3();

    if (all_pass)
        return 0;
    else
        return -1;
}
