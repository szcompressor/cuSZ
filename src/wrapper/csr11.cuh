/**
 * @file csr11.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HANDLE_SPARSITY11_CUH
#define CUSZ_WRAPPER_HANDLE_SPARSITY11_CUH

#include <driver_types.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "../../include/reducer.hh"

// clang-format off
template <typename F> struct cuszCUSPARSE;
template <> struct cuszCUSPARSE<float>  { const static cudaDataType type = CUDA_R_32F; };
template <> struct cuszCUSPARSE<double> { const static cudaDataType type = CUDA_R_64F; };
// clang-format on

namespace cusz {

template <typename T = float>
class CSR11 : public VirtualGatherScatter {
   public:
    using Origin                  = T;
    static const auto DEFAULT_LOC = cusz::LOC::DEVICE;

   private:
    // clang-format off
    uint8_t* pool_ptr;
    struct { unsigned int rowptr, colidx, values; } offset;
    struct { unsigned int rowptr, colidx, values, total; } nbyte;
    unsigned int workspace_nbyte, dump_nbyte;
    unsigned int m{0}, dummy_nnz{0}, nnz{0};
    float milliseconds{0.0};
    // clang-format on

    Capsule<int> rowptr;
    Capsule<int> colidx;
    Capsule<T>   values;

    // use when the real nnz is known
    void reconfigure_with_precise_nnz(int nnz);

#if CUDART_VERSION >= 11000
    void gather_CUDA11(T* in, unsigned int& dump_nbyte);
#elif CUDART_VERSION >= 10000
    void gather_CUDA10(T* in, unsigned int& dump_nbyte);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif

#if CUDART_VERSION >= 11000
    void scatter_CUDA11(T* out);
#elif CUDART_VERSION >= 10000
    void scatter_CUDA10(T* out);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif

    // TODO handle nnz == 0 otherwise
    unsigned int query_csr_bytelen() const
    {
        return sizeof(int) * (m + 1)  // rowptr
               + sizeof(int) * nnz    // colidx
               + sizeof(T) * nnz;     // values
    }

    void extract(uint8_t* _pool);

   public:
    // helper

    static uint32_t get_total_nbyte(uint32_t len, int nnz)
    {
        auto m = Reinterpret1DTo2D::get_square_size(len);
        return sizeof(int) * (m + 1) + sizeof(int) * nnz + sizeof(T) * nnz;
    }

    float get_time_elapsed() const { return milliseconds; }

    CSR11() = default;

    template <cusz::LOC FROM = cusz::LOC::DEVICE, cusz::LOC TO = cusz::LOC::HOST>
    CSR11& consolidate(uint8_t* dst);  //, cudaMemcpyKind direction = cudaMemcpyDeviceToHost);

    CSR11& decompress_set_nnz(unsigned int _nnz);

    void gather(T* in, unsigned int& nbyte_dump, int& out_nnz)  // removed memory pool
    {
#if CUDART_VERSION >= 11000
        gather_CUDA11(in, nbyte_dump);
#elif CUDART_VERSION >= 10000
        gather_CUDA10(in, nbyte_dump);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif
        out_nnz = this->nnz;
    }

    /**
     * @brief
     *
     * @param in
     * @param in_len
     * @param out_ptr nullable depending on impl.;
     * @param out_idx
     * @param out_val
     * @param nnz
     */
    void
    gather(T* in, uint32_t in_len, int* out_rowptr, int*& out_colidx, T*& out_val, int& out_nnz, uint32_t& nbyte_dump)
    {
        m = Reinterpret1DTo2D::get_square_size(in_len);

        if (out_rowptr) rowptr.template shallow_copy<DEFAULT_LOC>(out_rowptr);
        colidx.template shallow_copy<DEFAULT_LOC>(out_colidx);
        values.template shallow_copy<DEFAULT_LOC>(out_val);

#if CUDART_VERSION >= 11000
        // #pragma message("using gather-cuda11")
        gather_CUDA11(in, nbyte_dump);
#elif CUDART_VERSION >= 10000
        // #pragma message("using gather-cuda10")
        gather_CUDA10(in, nbyte_dump);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif
        out_nnz = this->nnz;
    }

    // only placeholding
    void scatter() {}
    void gather() {}

    void scatter(uint8_t* _pool, int nnz, T* out, uint32_t out_len)
    {
        m = Reinterpret1DTo2D::get_square_size(out_len);
        decompress_set_nnz(nnz);
        extract(_pool);

#if CUDART_VERSION >= 11000
        // #pragma message("using scatter-cuda11")
        scatter_CUDA11(out);
#elif CUDART_VERSION >= 10000
        // #pragma message("using scatter-cuda10")
        scatter_CUDA10(out);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif
    }
};

//
}  // namespace cusz

#endif