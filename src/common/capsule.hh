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
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <stdexcept>

#include "../utils/io.hh"
#include "../utils/timer.hh"
#include "definition.hh"

using cusz::OK;

template <typename T, bool USE_UNIFIED = false>
class Capsule {
   private:
    static const bool use_unified = USE_UNIFIED;

    template <cuszLOC LOC>
    void raise_error_if_misuse_unified()
    {
        static_assert(
            (LOC == cuszLOC::UNIFIED and USE_UNIFIED == true)           //
                or (LOC != cuszLOC::UNIFIED and USE_UNIFIED == false),  //
            "[Capsule] misused unified memory API");
    }

    template <cuszLOC LOC>
    void hostdevice_not_allowed()
    {
        static_assert(LOC != cuszLOC::HOST_DEVICE, "[Capsule] LOC at HOST_DEVICE not allowed");
    }

    std::string ERRSTR_BUILDER(std::string func, std::string msg)
    {  //
        return "[Capsule(\"" + name + "\")::" + func + "] " + msg;
    }

    std::string ERROR_UNDEFINED_BEHAVIOR(std::string func, std::string msg = "undefined behavior")
    {  //
        return ERRSTR_BUILDER(func, "undefined behavior");
    }

   public:
    using type = T;
    unsigned int len;

    std::string name;

    T* dptr;
    T* hptr;
    T* uniptr;

    template <cuszLOC LOC = cuszLOC::UNIFIED>
    T*& get()
    {
        raise_error_if_misuse_unified<LOC>();
        hostdevice_not_allowed<LOC>();

        if CONSTEXPR (LOC == cuszLOC::HOST)
            return hptr;
        else if (LOC == cuszLOC::DEVICE)
            return dptr;
        else if (LOC == cuszLOC::UNIFIED)
            return uniptr;
        else
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("get"));
    }

    template <cuszLOC LOC = cuszLOC::UNIFIED>
    T* set(T* ptr)
    {
        raise_error_if_misuse_unified<LOC>();
        hostdevice_not_allowed<LOC>();

        if CONSTEXPR (LOC == cuszLOC::HOST)
            hptr = ptr;
        else if (LOC == cuszLOC::DEVICE)
            dptr = ptr;
        else if (LOC == cuszLOC::UNIFIED)  // rare
            uniptr = ptr;
        else
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("set"));
    }

    Capsule() = default;

    Capsule(unsigned int _len, const std::string _str = std::string("<unnamed>")) : len(_len), name(_str) {}

    Capsule(const string _str) : name(_str){};

    Capsule(T* _h_in, T* _d_in, unsigned int _len) : hptr(_h_in), dptr(_d_in), len(_len) {}

    Capsule& set_name(std::string _str)
    {
        name = _str;
        return *this;
    }

    Capsule& set_len(unsigned int _len)
    {
        len = _len;
        return *this;
    }

    template <ALIGNDATA AD = ALIGNDATA::NONE>
    Capsule& get_len()
    {
        return Align::get_aligned_datalen<AD>(len);
    }

    template <cuszLOC LOC>
    Capsule& from_existing_on(T* in)
    {
        raise_error_if_misuse_unified<LOC>();
        hostdevice_not_allowed<LOC>();

        if (LOC == cuszLOC::HOST)
            hptr = in;
        else if (LOC == cuszLOC::DEVICE)
            dptr = in;
        else if (LOC == cuszLOC::UNIFIED)
            uniptr = in;
        else
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("from_existing_on"));

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
                if (not hptr) {  //
                    throw std::runtime_error(ERRSTR_BUILDER("from_fs_to", "hptr not set"));
                }
                io::read_binary_to_array<T>(fname, hptr, len);
            }
            else {
                throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("from_fs_to"));
            }
        }
        else if (DST == cuszLOC::DEVICE) {
            throw std::runtime_error(ERRSTR_BUILDER("to_fs_from", "to DEVICE not implemented"));
            // (VIA == cuszLOC::HOST)
            // (VIA == cuszLOC::NONE)
        }
        else if (DST == cuszLOC::UNIFIED) {
            if (not uniptr) {  //
                throw std::runtime_error(ERRSTR_BUILDER("to_fs_from", "uniptr not set"));
            }
            io::read_binary_to_array<T>(fname, uniptr, len);
        }
        else {
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("from_fs_to"));
        }

        auto z = hires::now();
        if (time) *time = static_cast<duration_t>(z - a).count();

        return *this;
    }

    template <cuszLOC SRC, cuszLOC VIA = cuszLOC::NONE>
    Capsule& to_fs_from(std::string fname)
    {
        if (SRC == cuszLOC::HOST) {
            if (not hptr) {  //
                throw std::runtime_error(ERRSTR_BUILDER("to_fs_from", "hptr not set"));
            }
            io::write_array_to_binary<T>(fname, hptr, len);
        }
        else if (SRC == cuszLOC::UNIFIED) {
            if (not uniptr) {  //
                throw std::runtime_error(ERRSTR_BUILDER("to_fs_from", "uniptr not set"));
            }
            io::write_array_to_binary<T>(fname, uniptr, len);
        }
        else {
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("to_fs_from"));
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

    /**
     * @brief
     *
     * @tparam LOC where to allocate: HOST, DEVICE, HOST_DEVICE, UNIFIED
     * @tparam AD alignment of data length; not reflecting the real datalen
     * @tparam AM alignment of underlying memory; not reflecting the real datalen
     * @tparam M mode, use with caution (easy to disable repo-wide)
     * @return Capsule& return *this for chained call
     */
    template <
        cuszLOC   LOC,  //
        ALIGNDATA AD = ALIGNDATA::NONE,
        ALIGNMEM  AM = ALIGNMEM::WARP128B,
        cuszDEV   M  = cuszDEV::DEV>
    Capsule& alloc()
    {
        OK::ALLOC<M>();
        raise_error_if_misuse_unified<LOC>();

        auto aligned_datalen    = Align::get_aligned_datalen<AD>(len);
        auto __memory_footprint = Align::get_aligned_nbyte<T>(aligned_datalen);

        if (LOC == cuszLOC::HOST) {
            cudaMallocHost(&hptr, __memory_footprint);
            cudaMemset(hptr, 0x00, __memory_footprint);
        }
        else if (LOC == cuszLOC::DEVICE) {
            cudaMalloc(&dptr, __memory_footprint);
            cudaMemset(dptr, 0x00, __memory_footprint);
        }
        else if (LOC == cuszLOC::HOST_DEVICE) {
            cudaMallocHost(&hptr, __memory_footprint);
            cudaMemset(hptr, 0x00, __memory_footprint);
            cudaMalloc(&dptr, __memory_footprint);
            cudaMemset(dptr, 0x00, __memory_footprint);
        }
        else if (LOC == cuszLOC::UNIFIED) {
            cudaMallocManaged(&uniptr, __memory_footprint);
            cudaMemset(uniptr, 0x00, __memory_footprint);
        }
        else {
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("alloc"));
        }

        return *this;
    }

    template <cuszLOC LOC, cuszDEV M = cuszDEV::DEV>
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
        else if (LOC == cuszLOC::UNIFIED) {
            cudaFree(uniptr);
        }
        else {
            throw std::runtime_error(ERROR_UNDEFINED_BEHAVIOR("free"));
        }

        return *this;
    }

   private:
    double maxval, minval, rng;

   public:
    double get_maxval() { return maxval; }
    double get_minval() { return minval; }
    double get_rng() { return rng; }

    Capsule& prescan(double& max_value, double& min_value, double& rng)
    {
        thrust::device_ptr<T> g_ptr;

        if (use_unified)
            g_ptr = thrust::device_pointer_cast(uniptr);
        else
            g_ptr = thrust::device_pointer_cast(dptr);

        // excluding padded
        auto max_el_loc = thrust::max_element(g_ptr, g_ptr + len);
        auto min_el_loc = thrust::min_element(g_ptr, g_ptr + len);

        max_value = *max_el_loc;
        min_value = *min_el_loc;
        rng       = max_value - min_value;

        return *this;
    }

    Capsule& prescan()
    {
        prescan(maxval, minval, rng);
        return *this;
    }
};

#endif