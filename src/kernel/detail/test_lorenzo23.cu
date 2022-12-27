/**
 * @file test_lorenzo23.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-23
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

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

template <typename T, typename EQ>
struct test_instance {
    T*  xdata;
    T*  outlier;
    EQ* eq;

    T*  h_xdata;
    T*  h_outlier;
    EQ* h_eq;

    void init(size_t len)
    {
        cudaMalloc(&xdata, sizeof(T) * len);
        cudaMalloc(&outlier, sizeof(T) * len);
        cudaMalloc(&eq, sizeof(EQ) * len);

        cudaMallocHost(&h_xdata, sizeof(T) * len);
        cudaMallocHost(&h_outlier, sizeof(T) * len);
        cudaMallocHost(&h_eq, sizeof(EQ) * len);
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

        cudaFreeHost(h_xdata);
        cudaFreeHost(h_outlier);
        cudaFreeHost(h_eq);
    }
};

template <typename T = float, typename EQ = uint16_t>
struct refactor_test_1d {
    using FP = T;

    T*     data;
    T*     h_data;
    size_t len           = 6480000;
    dim3   len3          = dim3(len, 1, 1);
    dim3   dummy_stride3 = dim3(0, 0, 0);
    int    radius        = 512;

    int grid_dim;

    double eb, ebx2, ebx2_r;

    refactor_test_1d(size_t _len) : len(_len) {}
    refactor_test_1d() {}

    void init_input_data_1d()
    {
        dim3 len3 = dim3(len, 1, 1);

        eb     = 1e-4;
        ebx2_r = 1 / (eb * 2);
        ebx2   = eb * 2;

        printf("[dbg] eb:\t%f\n", eb);
        printf("[dbg] ebx2:\t%f\n", ebx2);
        printf("[dbg] ebx2_r:\t%f\n", ebx2_r);

        cudaMalloc(&data, sizeof(T) * len);
        cudaMallocHost(&h_data, sizeof(T) * len);

        // parsz::testutils::cuda::rand_array<T>(data, len);

        auto fname = std::string(getenv("CESM"));
        read_binary_to_array(fname, h_data, len);

        // for (auto i = len / 2; i < len / 2 + 20; i++) cout << h_data[i] << "\t";
        // cout << endl;

        cudaMemcpy(data, h_data, sizeof(T) * len, cudaMemcpyHostToDevice);
    }

    void destroy_1d()
    {
        cudaFree(data);
        // cudaFreeHost(h_data);
    }

    template <int BLOCK, int SEQ>
    void test_1d()
    {
        grid_dim = (len - 1) / BLOCK + 1;

        // for (auto i = len / 2; i < len / 2 + 20; i++) cout << h_data[i] << "\t";
        // cout << endl;

        struct test_instance<T, EQ> ti1;
        struct test_instance<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len);

        before_refactor_1d<BLOCK, SEQ>(ti1);
        refactored_v0_1d<BLOCK, SEQ>(ti2);

        ti1.d2h(len);
        ti2.d2h(len);

        // cout << "\neq" << endl;
        // for (auto i = len / 2; i < len / 2 + 10; i++) printf("%d: (%d,%d)\n", i, ti1.h_eq[i], ti2.h_eq[i]);
        // cout << endl;

        // cout << "\noutlier" << endl;
        // for (int i = len / 2, count = 0; i < len / 2 + 1000 and count < 10; i++) {
        //     if (ti1.h_outlier[i] != 0) {
        //         printf("%d: (%f,%f)\n", i, ti1.h_outlier[i], ti2.h_outlier[i]);
        //         count++;
        //     }
        // }
        // cout << endl;

        // cout << "\nxdata" << endl;
        // for (auto i = len / 2; i < len / 2 + 10; i++) printf("%d: (%f,%f)\n", i, ti1.h_xdata[i], ti2.h_xdata[i]);
        // cout << endl;

        int count;

        count = 0;
        for (auto i = 0; i < len; i++) {
            auto eq1 = ti1.h_eq[i];
            auto eq2 = ti2.h_eq[i];
            if (eq1 != eq2 and count < 10) {
                cout << "eq not equal:\t" << eq1 << '\t' << eq2 << '\n';
                count++;
            }
        }

        count = 0;
        for (auto i = 0; i < len; i++) {
            auto outlier1 = ti1.h_outlier[i];
            auto outlier2 = ti2.h_outlier[i];
            if (outlier1 != outlier2 and count < 10) {
                cout << "eq not equal:\t" << outlier1 << '\t' << outlier2 << '\n';
                count++;
            }
        }

        count = 0;
        for (auto i = 0; i < len; i++) {
            auto xdata1 = ti1.h_xdata[i];
            auto xdata2 = ti2.h_xdata[i];
            if (xdata1 != xdata2 and count < 10) {
                cout << "xdata not equal:\t" << xdata1 << '\t' << xdata2 << '\n';
                count++;
            }
        }

        ti1.destroy();
        ti2.destroy();
    }

    template <int BLOCK, int SEQ>
    void before_refactor_1d(struct test_instance<T, EQ> ti)
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
    void refactored_v0_1d(struct test_instance<T, EQ> ti)
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

template <int BLOCK = 256, int SEQ = 8>
void test_inclusive_scan()
{
    using T  = float;
    using EQ = uint16_t;
    using FP = T;

    constexpr auto NTHREAD = BLOCK / SEQ;

    auto len  = BLOCK;
    auto ebx2 = 1;

    T*  data{nullptr};
    EQ* eq{nullptr};

    cudaMallocManaged(&data, sizeof(T) * len);
    cudaMallocManaged(&eq, sizeof(EQ) * len);
    cudaMemset(eq, 0x0, sizeof(EQ) * len);

    {
        cout << "original" << endl;
        for (auto i = 0; i < BLOCK; i++) data[i] = 1;

        cusz::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<1, NTHREAD>>>(data, eq, data, dim3(len, 1, 1), dim3(0, 0, 0), 0, ebx2);
        cudaDeviceSynchronize();

        for (auto i = 0; i < BLOCK; i++) cout << data[i] << " ";
        cout << "\n" << endl;
    }

    {
        cout << "refactored v0 (wave32)" << endl;
        for (auto i = 0; i < BLOCK; i++) data[i] = 1;

        parsz::cuda::__kernel::v0::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<1, NTHREAD>>>(eq, data, dim3(len, 1, 1), dim3(0, 0, 0), 0, ebx2, data);
        cudaDeviceSynchronize();

        for (auto i = 0; i < BLOCK; i++) cout << data[i] << " ";
        cout << "\n" << endl;
    }

    cudaFree(data);
    cudaFree(eq);
}

int main()
{
    struct refactor_test_1d<float, uint16_t> test {};
    test.init_input_data_1d();
    test.test_1d<256, 4>();
    test.destroy_1d();

    // test_inclusive_scan<256, 4>();
    // test_inclusive_scan<256, 8>();
    // test_inclusive_scan<512, 4>();
    // test_inclusive_scan<512, 8>();
    // test_inclusive_scan<1024, 4>();
    // test_inclusive_scan<1024, 8>();

    return 0;
}