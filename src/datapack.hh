/**
 * @file analysis_utils.hh
 * @author Jiannan Tian
 * @brief Simple data analysis (header)
 * @version 0.2.3
 * @date 2020-11-03
 * (create) 2020-11-03 (rev1) 2021-03-24
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef DATAPACK_H
#define DATAPACK_H

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#include <cuda_runtime.h>
#include <stdexcept>

enum class MODE { TEST, RELEASE };
enum class WHERE { HOST, DEVICE, BOTH };

template <typename T>
struct PartialData {
    using type = T;
    unsigned int len;
    T*           dptr;
    T*           hptr;

    PartialData() = default;

    PartialData(unsigned int _len) : len(_len) {}

    unsigned int nbyte() const { return len * sizeof(T); }

    // TODO really useful?
    struct PartialData& memset(unsigned char init = 0x0u)
    {
        cudaMemset(dptr, init, nbyte());
        return *this;
    }

    struct PartialData& h2d()
    {
        cudaMemcpy(dptr, hptr, nbyte(), cudaMemcpyHostToDevice);
        return *this;
    }
    struct PartialData& d2h()
    {
        cudaMemcpy(hptr, dptr, nbyte(), cudaMemcpyDeviceToHost);
        return *this;
    }
    template <MODE m, WHERE w>
    struct PartialData& alloc()
    {
        if (m != MODE::TEST) throw std::runtime_error("Impromptu allocation should only used when MODE::TEST.");

        if (w == WHERE::HOST) {
            cudaMallocHost(&hptr, nbyte());
            cudaMemset(hptr, 0x00, nbyte());
        }
        else if (w == WHERE::DEVICE) {
            cudaMalloc(&dptr, nbyte());
            cudaMemset(dptr, 0x00, nbyte());
        }
        else if (w == WHERE::BOTH) {
            cudaMallocHost(&hptr, nbyte());
            cudaMemset(hptr, 0x00, nbyte());
            cudaMalloc(&dptr, nbyte());
            cudaMemset(dptr, 0x00, nbyte());
        }
        else {
            throw std::runtime_error("Just ran into somewhere strange; not supposed to be here.");
        }

        return *this;
    }
    template <MODE m, WHERE w>
    struct PartialData& free()
    {
        if (m != MODE::TEST) throw std::runtime_error("Impromptu allocation should only used when MODE::TEST.");

        if (w == WHERE::HOST) {  //
            cudaFreeHost(hptr);
        }
        else if (w == WHERE::DEVICE) {
            cudaFree(dptr);
        }
        else if (w == WHERE::BOTH) {
            cudaFreeHost(hptr);
            cudaFree(dptr);
        }
        else {
            throw std::runtime_error("Just ran into somewhere strange; not supposed to be here.");
        }

        return *this;
    }
};

#endif