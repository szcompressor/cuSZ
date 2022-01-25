/**
 * @file ex_common.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-01-03
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef EX_COMMON_CUH
#define EX_COMMON_CUH

#include <cstdio>
#include <experimental/random>
#include <iostream>
#include <string>
using std::cerr;
using std::cout;
using std::endl;
using std::string;

#include "utils.hh"

using BYTE = uint8_t;
using SIZE = size_t;

#define PRINT_HEADER_ENTRY(SYM) \
    printf("header::%-*s: %d\n", 20, "entry[" #SYM "]", (*header).entry[COMPONENT::HEADER::SYM]);

template <typename T>
void echo_metric_gpu(T* d1, T* d2, size_t len)
{
    stat_t stat;
    verify_data_GPU<T>(&stat, d1, d2, len);
    analysis::print_data_quality_metrics<T>(&stat, 0, true);
}

template <typename T>
void echo_metric_cpu(T* _d1, T* _d2, size_t len, bool from_device = true)
{
    stat_t stat;
    T*     d1;
    T*     d2;
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
    analysis::verify_data<T>(&stat, d1, d2, len);
    analysis::print_data_quality_metrics<T>(&stat, 0, false);

    if (from_device) {
        if (d1) cudaFreeHost(d1);
        if (d2) cudaFreeHost(d2);
    }
}

/**
 * @brief Fill array with random intergers.
 *
 * @tparam T
 * @param a input array to change
 * @param len length
 * @param default_num the default number to fill the array
 * @param ratio the ratio of changed numbers in arrays
 */
template <typename T>
void gen_randint_array(T* a, SIZE len, int default_num, int ratio)
{
    auto fill = [&]() {  // fill base numbers
        for (auto i = 0; i < len; i++) { a[i] = default_num; }
    };
    auto rand_position   = [&]() { return std::experimental::randint(0ul, len - 1); };
    auto rand_signed_int = [&]() { return std::experimental::randint(-20, 20); };
    auto randomize       = [&]() {  // change 1/ratio
        for (auto i = 0; i < len / ratio; i++) a[rand_position()] += rand_signed_int();
    };
    // -----------------------------------------------------------------------------
    fill(), randomize();
}

/**
 * @brief Figure out error bound regarding the mode: abs or r2r.
 *
 * @tparam CAPSULE Capsule<T>
 * @param data input data
 * @param eb input error bound
 * @param adjusted_eb eb adjusted regarding mode
 * @param use_r2r if relative to value range
 */
template <typename CAPSULE>
void figure_out_eb(CAPSULE& data, double& eb, double& adjusted_eb, bool use_r2r)
{
    adjusted_eb = eb;

    if (use_r2r) {
        printf("using r2r mode...");
        auto rng = data.prescan().get_rng();
        adjusted_eb *= rng;
        printf("rng: %f\teb: %f\tadjusted eb: %f\n", rng, eb, adjusted_eb);
    }
    else {
        printf("using abs mode...");
        printf("eb: %f\n", eb);
    }
}

/**
 * @brief Barrier of device-wide or stream-based sync.
 *
 * @param stream
 */
void BARRIER(cudaStream_t stream = nullptr)
{
    if (not stream) {
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("device sync'ed\n");
    }
    else {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("stream sync'ed\n");
    }
}

template <typename UNCOMPRESSED>
void exp__prepare_data(
    UNCOMPRESSED** uncompressed,
    UNCOMPRESSED** decompressed,
    SIZE           len,
    int            base_number,
    int            non_base_portion,
    bool           destructive,
    UNCOMPRESSED** uncompressed_backup = nullptr)
{
    auto bytes = len * sizeof(UNCOMPRESSED);
    CHECK_CUDA(cudaMallocManaged(uncompressed, bytes));
    CHECK_CUDA(cudaMemset(*uncompressed, 0x0, bytes));
    gen_randint_array(*uncompressed, len, base_number, non_base_portion);

    CHECK_CUDA(cudaMallocManaged(decompressed, bytes));
    CHECK_CUDA(cudaMemset(*decompressed, 0x0, bytes));

    if (destructive)
        if (not uncompressed_backup) throw std::runtime_error("Destructive runtime must have data backed up.");
}

template <typename UNCOMPRESSED>
void exp__free(UNCOMPRESSED* uncompressed, UNCOMPRESSED* decompressed)
{
    cudaFree(uncompressed);
    cudaFree(decompressed);
}

template <typename UNCOMPRESSED>
void verify(UNCOMPRESSED*& uncompressed, UNCOMPRESSED*& decompressed, SIZE len)
{
    for (auto i = 0; i < len; i++) {
        auto un = uncompressed[i], de = decompressed[i];
        if (un != de) {
            printf("The decoding is corrupted, first not equal at %d: un[[ %d ]], de[[ %d ]]\n", i, (int)un, (int)de);
            return;
        }
    }
    printf("\nPASSED: The decoded/decompressed matches the original/uncompressed.\n");
}

template <typename DATA>
void verify_errorboundness(DATA*& origin, DATA*& reconstructed, double const eb, SIZE len)
{
    for (auto i = 0; i < len; i++) {
        auto un = (float)origin[i], de = (float)reconstructed[i];
        auto error = fabs(un - de);
        if (error >= eb) {
            printf(
                "Errorboundness is not guaranteed, first seen at %d: origin[[ %f ]], reconstructed[[ %f ]]\n", i, un,
                de);
            return;
        }
    }
    printf("\nPASSED: The predictor guarantees the errorboundness.\n");
}

#endif
