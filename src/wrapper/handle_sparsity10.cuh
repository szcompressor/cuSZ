/**
 * @file handle_sparsity10.cuh
 * @author Jiannan Tian
 * @brief (header) A high-level sparsity handling wrapper. Gather/scatter method to handle cuSZ prediction outlier.
 * @version 0.3
 * @date 2021-07-08
 * (created) 2020-09-10 (rev1) 2021-06-17 (rev2) 2021-07-08
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HANDLE_SPARSITY10_CUH
#define CUSZ_WRAPPER_HANDLE_SPARSITY10_CUH

#include <driver_types.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "../../include/reducer.hh"

namespace cusz {

template <typename Data = float>
class OutlierHandler10 : public OneCallGatherScatter {
   private:
    // clang-format off
    uint8_t* pool_ptr;
    struct { int * rowptr, *colidx; Data* values; } entry;
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

    void gather_CUDA10(float* in, unsigned int& dump_nbyte);

    void scatter_CUDA10(float* in_outlier);

    // TODO handle nnz == 0 otherwise
    unsigned int query_csr_bytelen() const
    {
        return sizeof(int) * (m + 1)  // rowptr
               + sizeof(int) * nnz    // colidx
               + sizeof(Data) * nnz;  // values
    }

    void archive(uint8_t* dst, int& nnz, cudaMemcpyKind direction = cudaMemcpyHostToDevice);

    void extract(uint8_t* _pool);

   public:
    // helper
    uint32_t get_total_nbyte() const { return nbyte.total; }

    float get_time_elapsed() const { return milliseconds; }

    // compression use
    OutlierHandler10(unsigned int _len, unsigned int* init_workspace_nbyte);

    void gather(Data* in, uint8_t* workspace, uint8_t* dump, unsigned int& dump_nbyte, int& out_nnz)
    {
        configure_workspace(workspace);
        gather_CUDA10(in, dump_nbyte);
        archive(dump, out_nnz);
    }

    // decompression use
    OutlierHandler10(unsigned int _len, unsigned int _nnz);

    // only placehoding
    void scatter() {}
    void gather() {}

    void scatter(uint8_t* _pool, Data* in_outlier)
    {
        extract(_pool);
        scatter_CUDA10(in_outlier);
    }
};

//
}  // namespace cusz

#endif