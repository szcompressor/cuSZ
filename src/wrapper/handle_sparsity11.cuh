/**
 * @file handle_sparsity11.cuh
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
class OutlierHandler11 : public OneCallGatherScatter {
   private:
    // clang-format off
    uint8_t* pool_ptr;
    struct { int * rowptr, *colidx; T* values; } entry;
    struct { unsigned int rowptr, colidx, values; } offset;
    struct { unsigned int rowptr, colidx, values, total; } nbyte;
    unsigned int workspace_nbyte, dump_nbyte;
    unsigned int m{0}, dummy_nnz{0}, nnz{0};
    float milliseconds{0.0};
    // clang-format on

    // set up memory pool
    void configure_workspace(uint8_t* _pool);

    // use when the real nnz is known
    void reconfigure_with_precise_nnz(int nnz);

    void gather_CUDA11(T* in, unsigned int& dump_nbyte);

    void scatter_CUDA11(T* out);

    // TODO handle nnz == 0 otherwise
    unsigned int query_csr_bytelen() const
    {
        return sizeof(int) * (m + 1)  // rowptr
               + sizeof(int) * nnz    // colidx
               + sizeof(T) * nnz;     // values
    }

    void archive(uint8_t* dst, int& nnz, cudaMemcpyKind direction = cudaMemcpyHostToDevice);

    void extract(uint8_t* _pool);

   public:
    // helper
    uint32_t get_total_nbyte() const { return nbyte.total; }

    float get_time_elapsed() const { return milliseconds; }

    // compression use
    OutlierHandler11(unsigned int _len, unsigned int* init_workspace_nbyte);

    void gather(T* in, uint8_t* workspace, uint8_t* dump, unsigned int& dump_nbyte, int& out_nnz)
    {
        configure_workspace(workspace);
        gather_CUDA11(in, dump_nbyte);
        archive(dump, out_nnz);
    }

    // decompression use
    OutlierHandler11(unsigned int _len, unsigned int _nnz);

    // only placehoding
    void scatter() {}
    void gather() {}

    void scatter(uint8_t* _pool, T* out)
    {
        extract(_pool);
        scatter_CUDA11(out);
    }
};

//
}  // namespace cusz

#endif