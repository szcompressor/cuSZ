/**
 * @file test_utils.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-26
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;

#include "../../../test/src/rand.hh"
#include "lorenzo.inl"
#include "lorenzo23.inl"

int get_num_SMs()
{
    int current_dev = 0;
    cudaSetDevice(current_dev);
    cudaDeviceProp dev_prop{};
    cudaGetDeviceProperties(&dev_prop, current_dev);

    return dev_prop.multiProcessorCount;
}

template <typename T>
void read_binary_to_array(const std::string& fname, T* _a, size_t dtype_len)
{
    std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << fname << std::endl;
        exit(1);
    }
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ifs.close();
}

template <typename FreeType>
void print_unconditional(FreeType* a, FreeType* b, size_t len, std::string what, int print_n = 10)
{
    auto count = 0;
    for (auto i = 0; i < len; i++) {
        auto aa = a[i], bb = b[i];
        if (count < print_n) {
            cout << what << ":\t" << aa << '\t' << bb << '\n';
            count++;
        }
    }
}

template <typename FreeType>
void print_non_zero(FreeType* a, FreeType* b, size_t len, std::string what, int print_n = 10)
{
    auto count = 0;
    for (auto i = 0; i < len; i++) {
        auto aa = a[i], bb = b[i];
        if (aa != 0 and count < print_n) {
            cout << what << ":\t" << aa << '\t' << bb << '\n';
            count++;
        }
    }
}

template <typename FreeType>
void print_when_not_equal(bool* equal, FreeType* a, FreeType* b, size_t len, std::string what, int print_n = 10)
{
    auto count = 0;
    for (auto i = 0; i < len; i++) {
        auto aa = a[i], bb = b[i];
        if (aa != bb and count < print_n) {
            cout << what << " not equal:\t" << aa << '\t' << bb << '\n';
            count++;
        }
    }

    if (equal) *equal = count == 0;
}

template <typename T, typename EQ>
struct TestPredictQuantize {
    T *xdata, *h_xdata;
    T *outlier, *h_outlier;

    EQ *eq, *h_eq;

    void init(size_t len)
    {
        cudaMalloc(&xdata, sizeof(T) * len);
        cudaMalloc(&outlier, sizeof(T) * len);
        cudaMalloc(&eq, sizeof(EQ) * len);

        cudaMemset(outlier, 0, sizeof(T) * len);
        cudaMemset(eq, 0, sizeof(EQ) * len);

        cudaMallocHost(&h_xdata, sizeof(T) * len);
        cudaMallocHost(&h_outlier, sizeof(T) * len);
        cudaMallocHost(&h_eq, sizeof(EQ) * len);
    }

    void init_managed(size_t len)
    {
        cudaMallocManaged(&xdata, sizeof(T) * len);
        cudaMallocManaged(&outlier, sizeof(T) * len);
        cudaMallocManaged(&eq, sizeof(EQ) * len);

        cudaMemset(outlier, 0, sizeof(T) * len);
        cudaMemset(eq, 0, sizeof(EQ) * len);
    }

    void d2h(size_t len)
    {
        cudaMemcpy(h_xdata, xdata, sizeof(T) * len, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_outlier, outlier, sizeof(T) * len, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eq, eq, sizeof(EQ) * len, cudaMemcpyDeviceToHost);
    };

    void destroy()
    {
        cudaFree(xdata);
        cudaFree(outlier);
        cudaFree(eq);

        if (h_xdata) cudaFreeHost(h_xdata);
        if (h_outlier) cudaFreeHost(h_outlier);
        if (h_eq) cudaFreeHost(h_eq);
    }
};

template <typename T = float, typename EQ = uint16_t>
struct RefactorTestFramework {
    using FP = T;

    T*     data;
    T*     h_data;
    size_t len           = 6480000;
    dim3   len3          = dim3(len, 1, 1);
    dim3   dummy_stride3 = dim3(0, 0, 0);
    int    radius        = 512;

    int grid_dim;

    double eb, ebx2, ebx2_r;

    RefactorTestFramework(size_t _len) : len(_len) {}
    RefactorTestFramework() {}

    RefactorTestFramework& set_eb(double _eb = 0.0)
    {
        eb     = eb != 0.0 ? eb = _eb : eb = 1e-4;
        ebx2_r = 1 / (eb * 2);
        ebx2   = eb * 2;

        return *this;
    }

    RefactorTestFramework& init_data_1d()
    {
        dim3 len3 = dim3(len, 1, 1);

        cudaMalloc(&data, sizeof(T) * len);
        cudaMallocHost(&h_data, sizeof(T) * len);

        // parsz::testutils::cuda::rand_array<T>(data, len);

        auto fname = std::string(getenv("CESM"));
        read_binary_to_array(fname, h_data, len);

        cudaMemcpy(data, h_data, sizeof(T) * len, cudaMemcpyHostToDevice);

        return *this;
    }

    RefactorTestFramework& destroy_1d()
    {
        cudaFree(data);
        if (h_data) cudaFreeHost(h_data);

        return *this;
    }

    template <int BLOCK, int SEQ>
    RefactorTestFramework& test1d_v0_against_origin()
    {
        grid_dim = (len - 1) / BLOCK + 1;

        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len);

        origin_1d<BLOCK, SEQ>(ti1);
        v0_1d<BLOCK, SEQ>(ti2);

        ti1.d2h(len);
        ti2.d2h(len);

        bool equal_eq, equal_outlier, equal_xdata;
        print_when_not_equal<EQ>(&equal_eq, ti1.h_eq, ti2.h_eq, len, "eq");
        print_when_not_equal<T>(&equal_outlier, ti1.h_outlier, ti2.h_outlier, len, "outlier");
        print_when_not_equal<T>(&equal_xdata, ti1.h_xdata, ti2.h_xdata, len, "xdata");

        if (equal_eq and equal_outlier and equal_xdata) printf("(all equal) PASS\n");

        ti1.destroy();
        ti2.destroy();

        return *this;
    }

    template <int BLOCK, int SEQ>
    void origin_1d(struct TestPredictQuantize<T, EQ> ti)
    {
        constexpr auto NTHREAD = BLOCK / SEQ;

        cusz::c_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<grid_dim, NTHREAD>>>(data, ti.eq, ti.outlier, len3, dummy_stride3, radius, ebx2_r);
        cudaDeviceSynchronize();

        cusz::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<grid_dim, NTHREAD>>>(ti.outlier, ti.eq, ti.xdata, len3, dummy_stride3, radius, ebx2);
        cudaDeviceSynchronize();
    }

    template <int BLOCK, int SEQ>
    void v0_1d(struct TestPredictQuantize<T, EQ> ti)
    {
        constexpr auto NTHREAD = BLOCK / SEQ;

        parsz::cuda::__kernel::v0::c_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<grid_dim, NTHREAD>>>(data, len3, dummy_stride3, radius, ebx2_r, ti.eq, ti.outlier);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<grid_dim, NTHREAD>>>(ti.eq, ti.outlier, len3, dummy_stride3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }
};