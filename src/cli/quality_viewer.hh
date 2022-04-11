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

#include "../analysis/verify.hh"
#include "../analysis/verify_gpu.cuh"
#include "../common/capsule.hh"
#include "../common/definition.hh"
#include "../header.hh"

namespace cusz {

const static auto HOST        = cusz::LOC::HOST;
const static auto DEVICE      = cusz::LOC::DEVICE;
const static auto HOST_DEVICE = cusz::LOC::HOST_DEVICE;

struct QualityViewer {
    template <typename Data>
    static void print_metrics(Stat* s, size_t compressed_bytes = 0, bool gpu_checker = false)
    {
        auto checker = (not gpu_checker) ? string("(using CPU checker)") : string("(using GPU checker)");
        auto bytes   = (s->len * sizeof(Data) * 1.0);

        auto println3 = [](const char* s, double n1, double n2, double n3) {
            printf("  %-10s %16.8g %16.8g %16.8g %16s\n", s, n1, n2, n3, "");
        };

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

        printhead("", "CR", "NRMSE", "corr-coeff", "PSNR");
        println("metrics", bytes / compressed_bytes, s->reduced.NRMSE, s->reduced.coeff, s->reduced.PSNR);

        printf("\n");
    };

    template <typename T>
    static void echo_metric_gpu(T* d1, T* d2, size_t len, size_t compressed_bytes = 0)
    {
        auto stat = new Stat;
        verify_data_GPU<T>(stat, d1, d2, len);
        print_metrics<T>(stat, compressed_bytes, true);
    }

    template <typename T>
    static void echo_metric_cpu(T* _d1, T* _d2, size_t len, size_t compressed_bytes = 0, bool from_device = true)
    {
        auto stat = new Stat;
        T*   d1;
        T*   d2;
        if (not from_device) {
            d1 = _d1;
            d2 = _d2;
        }
        else {
            printf("allocating tmp space for CPU verification\n");
            auto bytes = sizeof(T) * len;
            cudaMallocHost(&d1, bytes);
            cudaMallocHost(&d2, bytes);
            cudaMemcpy(d1, _d1, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(d2, _d2, bytes, cudaMemcpyDeviceToHost);
        }
        cusz::verify_data<T>(stat, d1, d2, len);
        print_metrics<T>(stat, compressed_bytes, false);

        if (from_device) {
            if (d1) cudaFreeHost(d1);
            if (d2) cudaFreeHost(d2);
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
        auto len             = (*header).get_len_uncompressed();
        auto compressd_bytes = (*header).get_filesize();

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
