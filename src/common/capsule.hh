/**
 * @file capsule.hh
 * @author Jiannan Tian
 * @brief Simple data analysis (header)
 * @version 0.2.3
 * @date 2020-11-03
 * (create) 2020-11-03 (rev1) 2021-03-24 (rev2) 2021-09-08
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CAPSULE_HH
#define CAPSULE_HH

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#include <cuda_runtime.h>
#include <stdexcept>

#include "../utils/io.hh"
#include "definition.hh"

using cusz::OK;

template <typename T>
class Capsule {
   public:
    using type = T;
    unsigned int len;

    T* dptr;
    T* hptr;

    Capsule() = default;

    Capsule(unsigned int _len) : len(_len) {}

    Capsule(T* _h_in, T* _d_in, unsigned int _len) : hptr(_h_in), dptr(_d_in), len(_len) {}

    Capsule& set_len(unsigned int _len)
    {
        len = _len;
        return *this;
    }

    template <cuszLOC LOC>
    Capsule& from_existing_on(T* in)
    {
        if (LOC == cuszLOC::HOST)
            hptr = in;
        else if (LOC == cuszLOC::DEVICE)
            dptr = in;
        else
            throw std::runtime_error("[Capsule::from_existing_on] undefined behavior");

        return *this;
    }

    /**
     * @brief specify single-source data
     *
     * @tparam DST the destination of loaded data
     * @tparam VIA the ephemeral/in transition
     * @param fname
     * @return Capsule&
     */
    template <cuszLOC DST, cuszLOC VIA = cuszLOC::NONE>
    Capsule& from_fs_to(std::string fname, double* time = nullptr)
    {
        auto a = hires::now();

        if (DST == cuszLOC::HOST) {
            if (VIA == cuszLOC::NONE) {
                if (not hptr) throw std::runtime_error("[Capsule::from_fs_to] hptr not set");
                io::read_binary_to_array<T>(fname, hptr, len);
            }
            else
                throw std::runtime_error("[Capsule::from_fs_to] undefined behavior");
        }
        else if (DST == cuszLOC::DEVICE) {
            throw std::runtime_error("[Capsule::from_fs_to] to DEVICE not implemented");
            // (VIA == cuszLOC::HOST)
            // (VIA == cuszLOC::NONE)
        }
        else {
            throw std::runtime_error("[Capsule::from_fs_to] undefined behavior");
        }

        auto z = hires::now();
        if (time) *time = static_cast<duration_t>(z - a).count();

        return *this;
    }

    template <cuszLOC SRC, cuszLOC VIA = cuszLOC::NONE>
    Capsule& to_fs_from(std::string fname)
    {
        if (SRC == cuszLOC::HOST) {
            if (not hptr) throw std::runtime_error("[Capsule::to_fs_from] hptr not set");
            io::write_array_to_binary<T>(fname, hptr, len);
        }
        else {
            throw std::runtime_error("[Capsule::to_fs_from] undefined behavior");
        }
        return *this;
    }

    unsigned int nbyte() const { return len * sizeof(T); }

    // TODO really useful?
    Capsule& memset(unsigned char init = 0x0u)
    {
        cudaMemset(dptr, init, nbyte());
        return *this;
    }

    Capsule& host2device()
    {
        cudaMemcpy(dptr, hptr, nbyte(), cudaMemcpyHostToDevice);
        return *this;
    }

    Capsule& device2host()
    {
        cudaMemcpy(hptr, dptr, nbyte(), cudaMemcpyDeviceToHost);
        return *this;
    }

    template <cuszDEV M, cuszLOC LOC>
    Capsule& alloc()
    {
        if (not OK::ALLOC<M>()) throw std::runtime_error("only cuszDEV::TEST or cuszDEV::DEV");

        if (LOC == cuszLOC::HOST) {
            cudaMallocHost(&hptr, nbyte());
            cudaMemset(hptr, 0x00, nbyte());
        }
        else if (LOC == cuszLOC::DEVICE) {
            cudaMalloc(&dptr, nbyte());
            cudaMemset(dptr, 0x00, nbyte());
        }
        else if (LOC == cuszLOC::HOST_DEVICE) {
            cudaMallocHost(&hptr, nbyte());
            cudaMemset(hptr, 0x00, nbyte());
            cudaMalloc(&dptr, nbyte());
            cudaMemset(dptr, 0x00, nbyte());
        }
        else {
            throw std::runtime_error("[Capsule::alloc] undefined behavior");
        }

        return *this;
    }

    template <cuszDEV M, cuszLOC LOC>
    Capsule& free()
    {
        if (not OK::FREE<M>()) throw std::runtime_error("only cuszDEV::TEST or cuszDEV::DEV");

        if (LOC == cuszLOC::HOST) {  //
            cudaFreeHost(hptr);
        }
        else if (LOC == cuszLOC::DEVICE) {
            cudaFree(dptr);
        }
        else if (LOC == cuszLOC::HOST_DEVICE) {
            cudaFreeHost(hptr);
            cudaFree(dptr);
        }
        else {
            throw std::runtime_error("[Capsule::free] undefined behavior");
        }

        return *this;
    }
};

#endif