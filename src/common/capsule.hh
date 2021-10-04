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
#include <driver_types.h>

#include <stdexcept>

#include "../utils/io.hh"
#include "../utils/timer.hh"
#include "definition.hh"

using cusz::OK;

template <typename T, bool USE_UNIFIED = false>
class Capsule {
   private:
    template <cuszLOC LOC>
    void raise_error_if_misuse_unified()
    {
        static_assert(
            (LOC == cuszLOC::UNIFIED and USE_UNIFIED == true)           //
                or (LOC != cuszLOC::UNIFIED and USE_UNIFIED == false),  //
            "[Capsule] misused unified memory API");
    }

   public:
    using type = T;
    unsigned int len;

    T* dptr;
    T* hptr;
    T* uniptr;

    template <cuszLOC LOC>
    T*& get()
    {
        raise_error_if_misuse_unified<LOC>();

        if CONSTEXPR (LOC == cuszLOC::HOST)
            return hptr;
        else if (LOC == cuszLOC::DEVICE)
            return dptr;
        else if (LOC == cuszLOC::UNIFIED)
            return uniptr;
        else
            throw std::runtime_error("[Capsule::get] undefined behavior");
    }

    template <cuszLOC LOC>
    T* set(T* ptr)
    {
        raise_error_if_misuse_unified<LOC>();

        if CONSTEXPR (LOC == cuszLOC::HOST)
            hptr = ptr;
        else if (LOC == cuszLOC::DEVICE)
            dptr = ptr;
        else if (LOC == cuszLOC::UNIFIED)  // rare
            uniptr = ptr;
        else
            throw std::runtime_error("[Capsule::set] undefined behavior");
    }

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
        raise_error_if_misuse_unified<LOC>();

        if (LOC == cuszLOC::HOST)
            hptr = in;
        else if (LOC == cuszLOC::DEVICE)
            dptr = in;
        else if (LOC == cuszLOC::UNIFIED)
            uniptr = in;
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
        else if (DST == cuszLOC::UNIFIED) {
            if (not uniptr) throw std::runtime_error("[Capsule::from_fs_to] uniptr not set");
            io::read_binary_to_array<T>(fname, uniptr, len);
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
        else if (SRC == cuszLOC::UNIFIED) {
            if (not uniptr) throw std::runtime_error("[Capsule::from_fs_to] uniptr not set");
            io::write_array_to_binary<T>(fname, uniptr, len);
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
        OK::ALLOC<M>();
        raise_error_if_misuse_unified<LOC>();

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
        else if (LOC == cuszLOC::UNIFIED) {
            cudaMallocManaged(&uniptr, nbyte());
            cudaMemset(uniptr, 0x00, nbyte());
        }
        else {
            throw std::runtime_error("[Capsule::alloc] undefined behavior");
        }

        return *this;
    }

    template <cuszDEV M, cuszLOC LOC>
    Capsule& free()
    {
        OK::FREE<M>();
        raise_error_if_misuse_unified<LOC>();

        if (LOC == cuszLOC::HOST)
            cudaFreeHost(hptr);
        else if (LOC == cuszLOC::DEVICE)
            cudaFree(dptr);
        else if (LOC == cuszLOC::HOST_DEVICE) {
            cudaFreeHost(hptr);
            cudaFree(dptr);
        }
        else if (LOC == cuszLOC::UNIFIED)
            cudaFree(uniptr);
        else
            throw std::runtime_error("[Capsule::free] undefined behavior");

        return *this;
    }
};

#endif