/**
 * @file quality_viewer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-09
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef QUALITY_VIEWER_HH
#define QUALITY_VIEWER_HH

#include <thrust/equal.h>

#include "../common/capsule.hh"
#include "../common/definition.hh"
#include "../header.h"
#include "verify.hh"
#include "verify_gpu.cuh"

namespace cusz {

const static auto HOST        = cusz::LOC::HOST;
const static auto DEVICE      = cusz::LOC::DEVICE;
const static auto HOST_DEVICE = cusz::LOC::HOST_DEVICE;

struct QualityViewer {
    template <typename Data>
    static void identical(cusz_execution_policy const policy, Data* d1, Data* d2, size_t const len)
    {
        bool result;

        if (policy == CUSZ_CPU) {  //
            result = thrust::equal(thrust::host, d1, d1 + len, d2);
        }
        else if (policy == CUSZ_GPU) {
            result = thrust::equal(thrust::device, d1, d1 + len, d2);
        }
        else {
            printf("Not a valid execution policy, exiting...\n");
            exit(-1);
        }

        if (result)
            cout << ">>>>  IDENTICAL." << endl;
        else
            cout << "!!!!  ERROR: NOT IDENTICAL." << endl;
    }

    template <typename Data>
    static void print_metrics_cross(cusz_stats* s, size_t compressed_bytes = 0, bool gpu_checker = false)
    {
        auto checker = (not gpu_checker) ? string("(using CPU checker)") : string("(using GPU checker)");
        auto bytes   = (s->len * sizeof(Data) * 1.0);

        auto println = [](const char* s, double n1, double n2, double n3, double n4) {
            printf("  %-10s %16.8g %16.8g %16.8g %16.8g\n", s, n1, n2, n3, n4);
        };
        auto printhead = [](const char* s1, const char* s2, const char* s3, const char* s4, const char* s5) {
            printf("  \e[1m\e[31m%-10s %16s %16s %16s %16s\e[0m\n", s1, s2, s3, s4, s5);
        };

        auto is_fp = std::is_same<Data, float>::value or std::is_same<Data, double>::value ? const_cast<char*>("yes")
                                                                                           : const_cast<char*>("no");
        printf("\nquality metrics %s:\n", checker.c_str());

        printhead("", "data-len", "data-byte", "fp-type?", "");
        printf("  %-10s %16zu %16lu %16s\n", "", s->len, sizeof(Data), is_fp);

        printhead("", "min", "max", "rng", "std");
        println("origin", s->odata.min, s->odata.max, s->odata.rng, s->odata.std);
        println("eb-lossy", s->xdata.min, s->xdata.max, s->xdata.rng, s->xdata.std);

        printhead("", "abs-val", "abs-idx", "pw-rel", "VS-RNG");
        println("max-error", s->max_err.abs, s->max_err.idx, s->max_err.pwrrel, s->max_err.rel);

        printhead("", "CR", "NRMSE", "cross-cor", "PSNR");
        println("metrics", bytes / compressed_bytes, s->reduced.NRMSE, s->reduced.coeff, s->reduced.PSNR);

        // printf("\n");
    };

    static void print_metrics_auto(double* lag1_cor, double* lag2_cor)
    {
        auto printhead = [](const char* s1, const char* s2, const char* s3, const char* s4, const char* s5) {
            printf("  \e[1m\e[31m%-10s %16s %16s %16s %16s\e[0m\n", s1, s2, s3, s4, s5);
        };

        printhead("", "lag1-cor", "lag2-cor", "", "");
        printf("  %-10s %16lf %16lf\n", "auto", *lag1_cor, *lag2_cor);
        printf("\n");
    };

    template <typename T>
    static void echo_metric_gpu(T* reconstructed, T* origin, size_t len, size_t compressed_bytes = 0)
    {
        // cross
        auto stat_x = new cusz_stats;
        verify_data_GPU<T>(stat_x, reconstructed, origin, len);
        print_metrics_cross<T>(stat_x, compressed_bytes, true);

        auto stat_auto_lag1 = new cusz_stats;
        verify_data_GPU<T>(stat_auto_lag1, origin, origin + 1, len - 1);
        auto stat_auto_lag2 = new cusz_stats;
        verify_data_GPU<T>(stat_auto_lag2, origin, origin + 2, len - 2);

        print_metrics_auto(&stat_auto_lag1->reduced.coeff, &stat_auto_lag2->reduced.coeff);
    }

    template <typename T>
    static void echo_metric_cpu(T* _d1, T* _d2, size_t len, size_t compressed_bytes = 0, bool from_device = true)
    {
        auto stat = new cusz_stats;
        T*   reconstructed;
        T*   origin;
        if (not from_device) {
            reconstructed = _d1;
            origin        = _d2;
        }
        else {
            printf("allocating tmp space for CPU verification\n");
            auto bytes = sizeof(T) * len;
            cudaMallocHost(&reconstructed, bytes);
            cudaMallocHost(&origin, bytes);
            cudaMemcpy(reconstructed, _d1, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(origin, _d2, bytes, cudaMemcpyDeviceToHost);
        }
        cusz::verify_data<T>(stat, reconstructed, origin, len);
        print_metrics_cross<T>(stat, compressed_bytes, false);

        auto stat_auto_lag1 = new cusz_stats;
        verify_data<T>(stat_auto_lag1, origin, origin + 1, len - 1);
        auto stat_auto_lag2 = new cusz_stats;
        verify_data<T>(stat_auto_lag2, origin, origin + 2, len - 2);

        print_metrics_auto(&stat_auto_lag1->reduced.coeff, &stat_auto_lag2->reduced.coeff);

        if (from_device) {
            if (reconstructed) cudaFreeHost(reconstructed);
            if (origin) cudaFreeHost(origin);
        }
    }

    template <typename T>
    static void load_origin(string const& fname, Capsule<T>& origin)
    {
        origin.template alloc<HOST_DEVICE>().template from_file<cusz::LOC::HOST>(fname);
    }

    template <typename T>
    static void view(header_t header, Capsule<T>& xdata, Capsule<T>& cmp, string const& compare)
    {
        auto len             = ConfigHelper::get_uncompressed_len(header);
        auto compressd_bytes = ConfigHelper::get_filesize(header);

        auto compare_on_gpu = [&]() {
            cmp.template alloc<HOST_DEVICE>().template from_file<HOST>(compare).host2device();
            echo_metric_gpu(xdata.dptr, cmp.dptr, len, compressd_bytes);
            cmp.template free<HOST_DEVICE>();
        };

        auto compare_on_cpu = [&]() {
            cmp.template alloc<HOST>().template from_file<HOST>(compare);
            xdata.device2host();
            echo_metric_cpu(xdata.hptr, cmp.hptr, len, compressd_bytes);
            cmp.template free<HOST>();
        };

        if (compare != "") {
            auto gb = 1.0 * sizeof(T) * len / 1e9;
            if (gb < 0.8)
                compare_on_gpu();
            else
                compare_on_cpu();
        }
    }
};

}  // namespace cusz

#endif
