/**
 * @file t-l23-utils.hh
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

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

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

    OutlierDescriptionGlobalMemory<T> outlier_desc;
    OutlierDescriptionGlobalMemory<T> h_outlier_desc;

    void init(size_t len, bool compaction = false, bool managed = false)
    {
        if (not managed) {
            cudaMalloc(&xdata, sizeof(T) * len);
            cudaMalloc(&eq, sizeof(EQ) * len);
            cudaMallocHost(&h_xdata, sizeof(T) * len);
            cudaMallocHost(&h_eq, sizeof(EQ) * len);
            cudaMemset(eq, 0, sizeof(EQ) * len);

            cudaMalloc(&outlier, sizeof(T) * len);
            cudaMallocHost(&h_outlier, sizeof(T) * len);
            cudaMemset(outlier, 0, sizeof(T) * len);

            if (compaction) {
                outlier_desc.allocate(len / 5);
                h_outlier_desc.allocate(len / 5, false /* not device */);
            }
        }
        else {
            cudaMallocManaged(&xdata, sizeof(T) * len);
            cudaMallocManaged(&eq, sizeof(EQ) * len);
            cudaMemset(eq, 0, sizeof(EQ) * len);

            cudaMallocManaged(&outlier, sizeof(T) * len);
            cudaMemset(outlier, 0, sizeof(T) * len);

            if (compaction) { outlier_desc.allocate_managed(len / 5); }
        }
    }

    void d2h(size_t len, bool compaction = false)
    {
        cudaMemcpy(h_xdata, xdata, sizeof(T) * len, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eq, eq, sizeof(EQ) * len, cudaMemcpyDeviceToHost);

        cudaMemcpy(h_outlier, outlier, sizeof(T) * len, cudaMemcpyDeviceToHost);

        if (compaction) {
            cudaMemcpy(h_outlier_desc.val, outlier_desc.val, sizeof(T) * len / 5, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outlier_desc.idx, outlier_desc.idx, sizeof(uint32_t) * len / 5, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outlier_desc.count, outlier_desc.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }
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
    size_t len  = 6480000;
    dim3   len3 = dim3(len, 1, 1);
    dim3   stride3;
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
        // TODO restrict 1D set here
        dim3 len3 = dim3(len, 1, 1);

        cudaMalloc(&data, sizeof(T) * len);
        cudaMallocHost(&h_data, sizeof(T) * len);

        // parsz::testutils::cuda::rand_array<T>(data, len);

        auto fname = std::string(getenv("CESM"));
        read_binary_to_array(fname, h_data, len);

        cudaMemcpy(data, h_data, sizeof(T) * len, cudaMemcpyHostToDevice);

        return *this;
    }

    RefactorTestFramework& init_data_2d()
    {
        dim3   len3    = dim3(3600, 1800, 1);
        dim3   stride3 = dim3(1, 3600, 1);
        size_t len     = 6480000;

        cudaMalloc(&data, sizeof(T) * len);
        cudaMallocHost(&h_data, sizeof(T) * len);

        auto fname = std::string(getenv("CESM"));
        read_binary_to_array(fname, h_data, len);

        cudaMemcpy(data, h_data, sizeof(T) * len, cudaMemcpyHostToDevice);

        return *this;
    }

    RefactorTestFramework& init_data_3d()
    {
        dim3   len3    = dim3(360, 180, 100);
        dim3   stride3 = dim3(1, 360, 180);
        size_t len     = 6480000;

        cudaMalloc(&data, sizeof(T) * len);
        cudaMallocHost(&h_data, sizeof(T) * len);

        auto fname = std::string(getenv("CESM"));
        read_binary_to_array(fname, h_data, len);

        cudaMemcpy(data, h_data, sizeof(T) * len, cudaMemcpyHostToDevice);

        return *this;
    }

    // destroy regardless of dimensionality
    RefactorTestFramework& destroy()
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
    RefactorTestFramework& test1d_v0compaction_against_origin()
    {
        grid_dim = (len - 1) / BLOCK + 1;

        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len, true);

        origin_1d<BLOCK, SEQ>(ti1);
        v0_compaction_1d<BLOCK, SEQ>(ti2);

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

    RefactorTestFramework& test2d_v0_against_origin()
    {
        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len);

        origin_2d(ti1, len3, stride3);
        v0_2d(ti2, len3, stride3);

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

    RefactorTestFramework& test2d_v0compaction_against_origin()
    {
        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len, true);

        origin_2d(ti1, len3, stride3);
        v0_compaction_2d(ti2, len3, stride3);

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

    RefactorTestFramework& test3d_v0_against_origin()
    {
        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len);

        origin_3d(ti1, len3, stride3);
        v0_3d(ti2, len3, stride3);

        ti1.d2h(len);
        ti2.d2h(len);

        bool equal_eq, equal_outlier, equal_xdata;
        print_when_not_equal<EQ>(&equal_eq, ti1.h_eq, ti2.h_eq, len, "eq");
        print_when_not_equal<T>(&equal_outlier, ti1.h_outlier, ti2.h_outlier, len, "outlier");
        print_when_not_equal<T>(&equal_xdata, ti1.h_xdata, ti2.h_xdata, len, "xdata");

        if (equal_eq and equal_outlier and equal_xdata) printf("3D v0 vs origin (all equal) PASS\n");

        ti1.destroy();
        ti2.destroy();

        return *this;
    }

    RefactorTestFramework& test3d_v0r1_against_origin()
    {
        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len);

        origin_3d(ti1, len3, stride3);
        v0r1_3d(ti2, len3, stride3);

        ti1.d2h(len);
        ti2.d2h(len);

        bool equal_eq, equal_outlier, equal_xdata;
        print_when_not_equal<EQ>(&equal_eq, ti1.h_eq, ti2.h_eq, len, "eq");
        print_when_not_equal<T>(&equal_outlier, ti1.h_outlier, ti2.h_outlier, len, "outlier");
        print_when_not_equal<T>(&equal_xdata, ti1.h_xdata, ti2.h_xdata, len, "xdata");

        if (equal_eq and equal_outlier and equal_xdata) printf("3D v0r1-shfl vs origin (all equal) PASS\n");

        ti1.destroy();
        ti2.destroy();

        return *this;
    }

    RefactorTestFramework& test3d_v0r1compaction_against_origin()
    {
        struct TestPredictQuantize<T, EQ> ti1;
        struct TestPredictQuantize<T, EQ> ti2;
        ti1.init(len);
        ti2.init(len, true);

        origin_3d(ti1, len3, stride3);
        v0r1_compaction_3d(ti2, len3, stride3);

        ti1.d2h(len);
        ti2.d2h(len);

        bool equal_eq, equal_outlier, equal_xdata;
        print_when_not_equal<EQ>(&equal_eq, ti1.h_eq, ti2.h_eq, len, "eq");
        print_when_not_equal<T>(&equal_outlier, ti1.h_outlier, ti2.h_outlier, len, "outlier");
        print_when_not_equal<T>(&equal_xdata, ti1.h_xdata, ti2.h_xdata, len, "xdata");

        if (equal_eq and equal_outlier and equal_xdata) printf("3D v0r1-shfl vs origin (all equal) PASS\n");

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

    template <int BLOCK, int SEQ>
    void v0_compaction_1d(struct TestPredictQuantize<T, EQ> ti)
    {
        constexpr auto NTHREAD = BLOCK / SEQ;

        parsz::cuda::__kernel::v0::compaction::c_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ, OutlierDescriptionGlobalMemory<T>>
            <<<grid_dim, NTHREAD>>>(data, len3, dummy_stride3, radius, ebx2_r, ti.eq, ti.outlier_desc);
        cudaDeviceSynchronize();

        cudaMemcpy(ti.h_outlier_desc.count, ti.outlier_desc.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        thrust::scatter(
            thrust::device,                                                         //
            ti.outlier_desc.val, ti.outlier_desc.val + (*ti.h_outlier_desc.count),  //
            ti.outlier_desc.idx,                                                    //
            ti.outlier /* full-size */);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
            <<<grid_dim, NTHREAD>>>(ti.eq, ti.outlier, len3, dummy_stride3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }

    void origin_2d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        // y-sequentiality == 8
        constexpr auto SUBLEN_2D = dim3(16, 16, 1);
        constexpr auto BLOCK_2D  = dim3(16, 2, 1);

        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };
        auto GRID_2D = divide3(len3, SUBLEN_2D);

        cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D>>>(data, ti.eq, ti.outlier, len3, leap3, radius, ebx2_r);
        cudaDeviceSynchronize();

        cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D>>>(ti.outlier, ti.eq, ti.xdata, len3, leap3, radius, ebx2);
        cudaDeviceSynchronize();
    }

    void v0_2d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        // y-sequentiality == 8
        constexpr auto SUBLEN_2D = dim3(16, 16, 1);
        constexpr auto BLOCK_2D  = dim3(16, 2, 1);

        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };
        auto GRID_2D = divide3(len3, SUBLEN_2D);

        parsz::cuda::__kernel::v0::c_lorenzo_2d1l<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D>>>(data, len3, leap3, radius, ebx2_r, ti.eq, ti.outlier);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_2d1l<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D>>>(ti.eq, ti.outlier, len3, leap3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }

    void v0_compaction_2d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        // y-sequentiality == 8
        constexpr auto SUBLEN_2D = dim3(16, 16, 1);
        constexpr auto BLOCK_2D  = dim3(16, 2, 1);

        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };
        auto GRID_2D = divide3(len3, SUBLEN_2D);

        parsz::cuda::__kernel::v0::compaction::c_lorenzo_2d1l<T, EQ, FP, OutlierDescriptionGlobalMemory<T>>
            <<<GRID_2D, BLOCK_2D>>>(data, len3, leap3, radius, ebx2_r, ti.eq, ti.outlier_desc);
        cudaDeviceSynchronize();

        cudaMemcpy(ti.h_outlier_desc.count, ti.outlier_desc.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        thrust::scatter(
            thrust::device,                                                         //
            ti.outlier_desc.val, ti.outlier_desc.val + (*ti.h_outlier_desc.count),  //
            ti.outlier_desc.idx,                                                    //
            ti.outlier /* full-size */);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_2d1l<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D>>>(ti.eq, ti.outlier, len3, leap3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }

    void origin_3d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };

        // y-sequentiality == 8
        constexpr auto SUBLEN_3D = dim3(32, 8, 8);
        constexpr auto BLOCK_3D  = dim3(32, 1, 8);
        auto           GRID_3D   = divide3(len3, SUBLEN_3D);

        cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, EQ, FP>
            <<<GRID_3D, BLOCK_3D>>>(data, ti.eq, ti.outlier, len3, leap3, radius, ebx2_r);
        cudaDeviceSynchronize();

        cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, EQ, FP>
            <<<GRID_3D, BLOCK_3D>>>(ti.outlier, ti.eq, ti.xdata, len3, leap3, radius, ebx2);
        cudaDeviceSynchronize();
    }

    void v0_3d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };

        // y-sequentiality == 8
        constexpr auto SUBLEN_3D = dim3(32, 8, 8);
        constexpr auto BLOCK_3D  = dim3(32, 1, 8);
        auto           GRID_3D   = divide3(len3, SUBLEN_3D);

        parsz::cuda::__kernel::v0::c_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, BLOCK_3D>>>(data, len3, leap3, radius, ebx2_r, ti.eq, ti.outlier);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, BLOCK_3D>>>(ti.eq, ti.outlier, len3, leap3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }

    void v0r1_3d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };

        // y-sequentiality == 8
        constexpr auto SUBLEN_3D = dim3(32, 8, 8);
        auto           GRID_3D   = divide3(len3, SUBLEN_3D);

        parsz::cuda::__kernel::v0::r1_shfl::c_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, dim3(32, 8, 1)>>>(data, len3, leap3, radius, ebx2_r, ti.eq, ti.outlier);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, dim3(32, 1, 8)>>>(ti.eq, ti.outlier, len3, leap3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }

    void v0r1_compaction_3d(struct TestPredictQuantize<T, EQ> ti, dim3 len3, dim3 leap3)
    {
        auto divide3 = [](dim3 len, dim3 sublen) {
            return dim3(
                (len.x - 1) / sublen.x + 1,  //
                (len.y - 1) / sublen.y + 1,  //
                (len.z - 1) / sublen.z + 1);
        };

        // y-sequentiality == 8
        constexpr auto SUBLEN_3D = dim3(32, 8, 8);
        auto           GRID_3D   = divide3(len3, SUBLEN_3D);

        parsz::cuda::__kernel::v0::r1_shfl::compaction::c_lorenzo_3d1l<T, EQ, FP, OutlierDescriptionGlobalMemory<T>>
            <<<GRID_3D, dim3(32, 8, 1)>>>(data, len3, leap3, radius, ebx2_r, ti.eq, ti.outlier_desc);
        cudaDeviceSynchronize();

        cudaMemcpy(ti.h_outlier_desc.count, ti.outlier_desc.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        thrust::scatter(
            thrust::device,                                                         //
            ti.outlier_desc.val, ti.outlier_desc.val + (*ti.h_outlier_desc.count),  //
            ti.outlier_desc.idx,                                                    //
            ti.outlier /* full-size */);
        cudaDeviceSynchronize();

        parsz::cuda::__kernel::v0::x_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, dim3(32, 1, 8)>>>(ti.eq, ti.outlier, len3, leap3, radius, ebx2, ti.xdata);
        cudaDeviceSynchronize();
    }
};
