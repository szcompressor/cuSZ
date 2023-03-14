/**
 * @file test_core_serial.cc
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

#include "kernel/detail/lorenzo_proto.inl"

using std::cout;
using std::endl;
namespace proto = psz::cuda::__kernel::prototype;

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
    T* input;
    cudaMalloc(&input, sizeof(T) * len);
    cudaMemcpy(input, h_input, sizeof(T) * len, cudaMemcpyHostToDevice);

    EQ *eq, *h_eq;
    cudaMalloc(&eq, sizeof(EQ) * len), cudaMallocHost(&h_eq, sizeof(EQ) * len);
    cudaMemset(eq, 0, sizeof(EQ) * len), memset(h_eq, 0, sizeof(EQ) * len);

    T* outlier;
    cudaMalloc(&outlier, sizeof(T) * len), cudaMemset(outlier, 0, sizeof(T) * len);

    auto radius = 512;

    if (dim == 1)
        proto::c_lorenzo_1d1l<T><<<t1d_grid_dim, t1d_block_dim>>>(input, len3, stride3, radius, 1, eq, outlier);
    else if (dim == 2)
        proto::c_lorenzo_2d1l<T><<<t2d_grid_dim, t2d_block_dim>>>(input, len3, stride3, radius, 1, eq, outlier);
    else if (dim == 3)
        proto::c_lorenzo_3d1l<T><<<t3d_grid_dim, t3d_block_dim>>>(input, len3, stride3, radius, 1, eq, outlier);
    cudaDeviceSynchronize();

    cudaMemcpy(h_eq, eq, sizeof(EQ) * len, cudaMemcpyDeviceToHost);

    // for (auto i = 0; i < len; i++) { cout << h_eq[i] << endl; }

    bool ok = true;
    for (auto i = 0; i < len; i++) {
        // subject to change according to the algorithm
        if (h_eq[i] - radius != h_expected_output[i]) {
            ok = false;
            break;
        }
    }
    cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

    cudaFree(input);
    cudaFree(eq);
    cudaFree(outlier);

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

    EQ *input, *h_input;
    cudaMalloc(&input, sizeof(EQ) * len), cudaMallocHost(&h_input, sizeof(EQ) * len);
    cudaMemset(input, 0, sizeof(EQ) * len), memset(h_input, 0, sizeof(EQ) * len);
    for (auto i = 0; i < len; i++) h_input[i] = _h_input[i] + radius;
    cudaMemcpy(input, h_input, sizeof(EQ) * len, cudaMemcpyHostToDevice);

    T *xdata, *h_xdata;
    cudaMalloc(&xdata, sizeof(T) * len);
    cudaMemset(xdata, 0, sizeof(T) * len);
    cudaMallocHost(&h_xdata, sizeof(T) * len);
    memset(h_xdata, 0, sizeof(T) * len);

    if (dim == 1)
        proto::x_lorenzo_1d1l<T><<<t1d_grid_dim, t1d_block_dim>>>(
            const_cast<EQ*>(input), xdata /* outlier */, len3, stride3, radius, 1, xdata);
    else if (dim == 2)
        proto::x_lorenzo_2d1l<T><<<t2d_grid_dim, t2d_block_dim>>>(
            const_cast<EQ*>(input), xdata /* outlier */, len3, stride3, radius, 1, xdata);
    else if (dim == 3)
        proto::x_lorenzo_3d1l<T><<<t3d_grid_dim, t3d_block_dim>>>(
            const_cast<EQ*>(input), xdata /* outlier */, len3, stride3, radius, 1, xdata);
    else {
        throw std::runtime_error("must be 1, 2, or 3D.");
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_xdata, xdata, sizeof(T) * len, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (auto i = 0; i < len; i++) {
        if (h_xdata[i] != h_expected_output[i]) {
            ok = false;
            break;
        }
    }
    cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

    cudaFree(input);
    cudaFree(xdata);
    cudaFreeHost(h_input);
    cudaFreeHost(h_xdata);

    return ok;
}

bool test3(int dim, T const* h_input, size_t len, dim3 len3, dim3 stride3, std::string funcname)
{
    T* input;
    cudaMalloc(&input, sizeof(T) * len);
    cudaMemcpy(input, h_input, sizeof(T) * len, cudaMemcpyHostToDevice);

    EQ *eq, *h_eq;
    cudaMalloc(&eq, sizeof(EQ) * len);
    cudaMemset(eq, 0, sizeof(EQ) * len);
    cudaMallocHost(&h_eq, sizeof(EQ) * len);
    memset(h_eq, 0, sizeof(EQ) * len);

    T* outlier;
    cudaMalloc(&outlier, sizeof(T) * len);
    cudaMemset(outlier, 0, sizeof(T) * len);

    T *xdata, *h_xdata;
    cudaMalloc(&xdata, sizeof(T) * len);
    cudaMemset(xdata, 0, sizeof(T) * len);
    cudaMallocHost(&h_xdata, sizeof(T) * len);
    memset(h_xdata, 0, sizeof(T) * len);

    auto radius = 512;

    auto eb     = 1e-2;
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / (eb * 2);

    if (dim == 1) {
        proto::c_lorenzo_1d1l<T><<<t1d_grid_dim, t1d_block_dim>>>(input, len3, stride3, radius, ebx2_r, eq, outlier);
        cudaDeviceSynchronize();
        proto::x_lorenzo_1d1l<T>
            <<<t1d_grid_dim, t1d_block_dim>>>(eq, xdata /* outlier */, len3, stride3, radius, ebx2, xdata);
        cudaDeviceSynchronize();
    }
    else if (dim == 2) {
        proto::c_lorenzo_2d1l<T><<<t2d_grid_dim, t2d_block_dim>>>(input, len3, stride3, radius, ebx2_r, eq, outlier);
        cudaDeviceSynchronize();
        proto::x_lorenzo_2d1l<T>
            <<<t2d_grid_dim, t2d_block_dim>>>(eq, xdata /* outlier */, len3, stride3, radius, ebx2, xdata);
        cudaDeviceSynchronize();
    }
    else if (dim == 3) {
        proto::c_lorenzo_3d1l<T><<<t3d_grid_dim, t3d_block_dim>>>(input, len3, stride3, radius, ebx2_r, eq, outlier);
        cudaDeviceSynchronize();
        proto::x_lorenzo_3d1l<T>
            <<<t3d_grid_dim, t3d_block_dim>>>(eq, xdata /* outlier */, len3, stride3, radius, ebx2, xdata);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_xdata, xdata, sizeof(EQ) * len, cudaMemcpyDeviceToHost);

    // for (auto i = 0; i < len; i++) { cout << h_xdata[i] << endl; }

    bool ok = true;
    for (auto i = 0; i < len; i++) {
        if (h_xdata[i] != h_input[i]) {
            ok = false;
            break;
        }
    }
    cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

    cudaFree(input);
    cudaFree(eq);
    cudaFree(outlier);
    cudaFree(xdata);
    cudaFreeHost(h_xdata);

    return ok;
}

int main()
{
    auto all_pass = true;

    all_pass = all_pass and test1(1, t1d_in, t1d_len, t1d_len3, t1d_stride3, t1d_comp_out, "standalone cl1d1l");
    all_pass = all_pass and test1(2, t2d_in, t2d_len, t2d_len3, t2d_stride3, t2d_comp_out, "standalone cl2d1l");
    all_pass = all_pass and test1(3, t3d_in, t3d_len, t3d_len3, t3d_stride3, t3d_comp_out, "standalone cl3d1l");

    all_pass = all_pass and test2(1, t1d_eq, t1d_len, t1d_len3, t1d_stride3, t1d_decomp_out, "standalone xl1d1l");
    all_pass = all_pass and test2(2, t2d_eq, t2d_len, t2d_len3, t2d_stride3, t2d_decomp_out, "standalone xl2d1l");
    all_pass = all_pass and test2(3, t3d_eq, t3d_len, t3d_len3, t3d_stride3, t3d_decomp_out, "standalone xl3d1l");

    all_pass = all_pass and test3(1, t1d_in, t1d_len, t1d_len3, t1d_stride3, "lorenzo_1d1l");
    all_pass = all_pass and test3(2, t2d_in, t2d_len, t2d_len3, t2d_stride3, "lorenzo_2d1l");
    all_pass = all_pass and test3(3, t3d_in, t3d_len, t3d_len3, t3d_stride3, "lorenzo_3d1l");

    if (all_pass)
        return 0;
    else
        return -1;
}
