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

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "type_trait.hh"
#include "utils/io.hh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

enum class transfer { h2d, d2h, fs2h, h2fs, fs2d, d2fs, d2d, h2h };
enum class space { host, device };

// TODO move elsewhere
template <typename T>
class DataPack {
    T* _hptr{nullptr};
    T* _dptr{nullptr};

    size_t padding{};

    std::string name{"<unnamed>"};

    std::string print_name() const { return "\e[1m[array::" + name + "]\e[0m "; }

    struct {
#if __cplusplus > 201703L
        bool hptr_allocated : 1                            = false;
        bool hptr_from_external_do_not_free_on_destroy : 1 = false;
        bool dptr_allocated : 1                            = false;
        bool dptr_from_external_do_not_free_on_destroy : 1 = false;
        bool length_valid : 1                              = false;
        bool padding_added : 1                             = false;
        bool pseudo_matrix : 1                             = false;
#else
        bool hptr_allocated : 1;
        bool hptr_from_external_do_not_free_on_destroy : 1;
        bool dptr_allocated : 1;
        bool dptr_from_external_do_not_free_on_destroy : 1;
        bool length_valid : 1;
        bool padding_added : 1;
        bool pseudo_matrix : 1;
#endif
    } property;

   public:
    // type access
    using Type = T;
    bool HptrAllocated() const { return property.hptr_allocated; }
    bool DptrAllocated() const { return property.dptr_allocated; }

   public:
    // variable accessor
    size_t len{}, sqrt_ceil{}, pseudo_matrix_size{};
    T*     hptr() const { return _hptr; }
    T*     dptr() const { return _dptr; }
    T*     safe_hptr() const
    {
        if (not _hptr) throw std::runtime_error(print_name() + "safe_hptr() hptr not set.");
        return _hptr;
    }
    T* safe_dptr() const
    {
        if (not _dptr) throw std::runtime_error(print_name() + "safe_dptr() hptr not set.");
        return _dptr;
    }
    size_t Len() const { return len; }
    size_t NumBytes() const { return sizeof(T) * len; }

    template <space s>
    T* begin() const
    {
        if CONSTEXPR (s == space::host) { return _hptr; }
        else if CONSTEXPR (s == space::device) {
            return _dptr;
        }
    }

    template <space s>
    T* end() const
    {
        if CONSTEXPR (s == space::host) { return _hptr + len; }
        else if CONSTEXPR (s == space::device) {
            return _dptr + len;
        }
    }

   public:
    explicit DataPack(std::string name_ = "", size_t _len = 0)
    {
#if __cplusplus <= 201703L
        property.hptr_allocated                            = false;
        property.dptr_allocated                            = false;
        property.hptr_from_external_do_not_free_on_destroy = false;
        property.dptr_from_external_do_not_free_on_destroy = false;
        property.length_valid                              = false;
        property.padding_added                             = false;
        property.pseudo_matrix                             = false;
#endif
        name = std::move(name_);

        if (_len != 0) { property.length_valid = true, this->len = _len; }
    };

    ~DataPack()
    {
        if (property.hptr_allocated and not property.hptr_from_external_do_not_free_on_destroy) {
            cudaFreeHost(_hptr);
            _hptr = nullptr;
        }
        if (property.dptr_allocated and not property.dptr_from_external_do_not_free_on_destroy) {
            cudaFree(_dptr);
            _dptr = nullptr;
        }
    };

   public:
    DataPack& SetLen(size_t _len, bool pseudo_sqmat_padding = false)
    {
        if (_len == 0) { throw std::runtime_error(print_name() + "SetLen() Input length is 0."); }
        property.length_valid = true, this->len = _len;

        if (pseudo_sqmat_padding) {
            sqrt_ceil              = static_cast<size_t>(ceil(sqrt(len)));  // row-major mxn matrix
            pseudo_matrix_size     = sqrt_ceil * sqrt_ceil;
            padding                = pseudo_matrix_size - len;
            property.pseudo_matrix = true;
        }
        return *this;
    }

    template <space s>
    DataPack& Memset(uint8_t filling_val = 0x00)
    {
        if CONSTEXPR (s == space::host) {
            if (not property.hptr_allocated)
                throw std::runtime_error(print_name() + "Memset<space::host>() Host array not allocated.");
            memset(_hptr, filling_val, (len + padding) * sizeof(T));
        }
        else if CONSTEXPR (s == space::device) {
            if (not property.dptr_allocated)
                throw std::runtime_error(print_name() + "Memset<space::device>() Device array not allocated.");
            cudaMemset(_dptr, filling_val, (len + padding) * sizeof(T));
        }

        return *this;
    }

    // TODO deal with padding_
    DataPack& AllocHostSpace(size_t padding_ = 0, uint8_t filling_val = 0x00)
    {
        // validate allocation
        if (property.hptr_allocated)
            throw std::runtime_error(print_name() + "AllocHostSpace() Host array has already been allocated.");
        if (not property.length_valid) throw std::runtime_error(print_name() + "AllocHostSpace() Length is not valid.");

        // check padding
        if ((padding_ != 0) and (property.padding_added)) {
            throw std::runtime_error(print_name() + "AllocHostSpace() Different/redundant padding specified.");
        }
        else {
            padding                = padding_;
            property.padding_added = true;
        }

        cudaMallocHost(&_hptr, (len + padding) * sizeof(T));
        memset(_hptr, filling_val, (len + padding) * sizeof(T));
        property.hptr_allocated = true;

        return *this;
    }

    // TODO deal with padding
    DataPack& FreeHostSpace()
    {
        if (not property.hptr_allocated) {
            throw std::runtime_error(print_name() + "FreeHostSpace() Host array has NOT been allocated.");
        }
        else {
            cudaFreeHost(_hptr);
            _hptr                   = nullptr;
            property.hptr_allocated = false;
        }
        return *this;
    }

    DataPack& AllocDeviceSpace(size_t padding_ = 0, uint8_t filling_val = 0x00)
    {
        // validate allocation
        if (property.dptr_allocated)
            throw std::runtime_error(print_name() + "AllocDeviceSpace() Device array has already been allocated.");
        if (not property.length_valid)
            throw std::runtime_error(print_name() + "AllocDeviceSpace() Length is not valid.");

        // check padding
        if ((padding_ != 0) and (property.padding_added)) {
            throw std::runtime_error(print_name() + "AllocDeviceSpace() Different/redundant padding specified.");
        }
        else {
            padding                = padding_;
            property.padding_added = true;
        }

        cudaMalloc(&_dptr, (len + padding) * sizeof(T));
        cudaMemset(_dptr, filling_val, (len + padding) * sizeof(T));
        property.dptr_allocated = true;
        return *this;
    }

    DataPack& FreeDeviceSpace()
    {
        if (not property.dptr_allocated) {
            throw std::runtime_error(print_name() + "FreeDeviceSpace() Device array has NOT been allocated.");
        }
        else {
            cudaFree(_dptr);
            _dptr                   = nullptr;
            property.dptr_allocated = false;
        }
        return *this;
    }

    DataPack& SetHostSpace(T* h_ext)
    {
        if (h_ext == nullptr) {
            throw std::runtime_error(print_name() + "SetHostSpace() External host array is null.");
        }
        else {
            _hptr                   = h_ext;
            property.hptr_allocated = true, property.hptr_from_external_do_not_free_on_destroy = true;
        }
        return *this;
    }

    DataPack& SetDeviceSpace(T* d_ext)
    {
        if (d_ext == nullptr) {
            throw std::runtime_error(print_name() + "SetDeviceSpace() External host array is null.");
        }
        else {
            _dptr                   = d_ext;
            property.dptr_allocated = true, property.dptr_from_external_do_not_free_on_destroy = true;
        }
        return *this;
    }

    DataPack& Note(placeholder p) { return *this; }

    template <transfer t>
    DataPack& Move(string filename = "")
    {
        if (t == transfer::h2d or t == transfer::d2h) {  //
            if (not property.length_valid) throw std::runtime_error(print_name() + "Move() Length not valid.");
            if (not property.hptr_allocated) throw std::runtime_error(print_name() + "Move() hptr not allocated.");
            if (not property.dptr_allocated) throw std::runtime_error(print_name() + "Move() dptr not allocated.");
            // move
            if (t == transfer::h2d) cudaMemcpy(_dptr, _hptr, sizeof(T) * len, cudaMemcpyHostToDevice);
            if (t == transfer::d2h) cudaMemcpy(_hptr, _dptr, sizeof(T) * len, cudaMemcpyDeviceToHost);
        }
        else if (t == transfer::h2fs or t == transfer::fs2h) {
            if (not property.length_valid) throw std::runtime_error(print_name() + "Move() Length not valid.");
            if (not property.hptr_allocated) throw std::runtime_error(print_name() + "Move() hptr not allocated.");

            if (t == transfer::h2fs) io::WriteArrayToBinary(filename, _hptr, len);
            if (t == transfer::fs2h) io::ReadBinaryToArray(filename, _hptr, len);
        }
        else if (t == transfer::d2fs or t == transfer::fs2d) {
            throw std::runtime_error("Move() d2fs not implemented");
        }
        return *this;
    };

    // TODO use static assert
    template <typename T_, transfer t>
    DataPack& CopyContentTo(DataPack<T_>* target)
    {
        // check type
        if (std::is_same<Type, T_>::value)
            throw std::runtime_error(print_name() + "CopyContentTo() Target has different type.");

        // ignore padding difference if any
        // TODO corner case: if padding is set to some other filling value
        if (this->Len() != target->Len())
            throw std::runtime_error(print_name() + "CopyContentTo() Target has different length (ignore padding).");

        if CONSTEXPR (t == transfer::h2h) {
            if (this->HptrAllocated() and target->HptrAllocated())
                std::copy(this->hptr(), this->hptr() + this->Len(), target->hptr());
            else
                throw std::runtime_error(print_name() + "CopyContentTo() Host allocation error in one or both arrays.");
        }
        else if CONSTEXPR (t == transfer::d2d) {
            if (this->DptrAllocated() and target->DptrAllocated())
                cudaMemcpy(target->dptr(), this->dptr(), this->Len() * sizeof(Type), cudaMemcpyDeviceToDevice);
            else
                throw std::runtime_error(
                    print_name() + "CopyContentTo() Device allocation error in one or both arrays.");
        }
        else if CONSTEXPR (t == transfer::h2d) {
            if (this->HptrAllocated() and target->DptrAllocated())
                cudaMemcpy(target->hptr(), this->dptr(), this->Len() * sizeof(Type), cudaMemcpyHostToDevice);
            throw std::runtime_error(
                print_name() + "CopyContentTo() Requiring (src/this) hspace allocated and (target) dspace allocated.");
        }
        else if CONSTEXPR (t == transfer::d2h) {
            if (this->DptrAllocated() and target->HptrAllocated())
                cudaMemcpy(target->dptr(), this->hptr(), this->Len() * sizeof(Type), cudaMemcpyDeviceToHost);
            throw std::runtime_error(
                print_name() + "CopyContentTo() Requiring (src/this) hspace allocated and (target) dspace allocated.");
        }
        else {
            throw std::runtime_error(print_name() + "CopyContentTo() Must be transfer::{d,h}2{d,h}.");
        }

        return *this;
    }
};

#endif