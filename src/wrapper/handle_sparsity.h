/**
 * @file handle_sparsity.h
 * @author Jiannan Tian
 * @brief (header) A high-level sparsity handling wrapper. Gather/scatter method to handle cuSZ prediction outlier.
 * @version 0.3
 * @date 2021-07-08
 * (created) 2020-09-10 (rev1) 2021-06-17 (rev2) 2021-07-08
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HANDLE_SPARSITY_H
#define CUSZ_WRAPPER_HANDLE_SPARSITY_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

template <typename Data>
unsigned int get_compression_workspace_nbyte(unsigned int len)
{
    auto m        = static_cast<size_t>(ceil(sqrt(len)));
    auto init_nnz = len / 10;

    return sizeof(int) * (m + 1) +   // rowptr
           sizeof(int) * init_nnz +  // colidx
           sizeof(Data) * init_nnz;  // values
}

template <typename Data = float>
class OutlierHandler {
   public:
    struct {
        uint8_t* ptr;

        struct {
            int * rowptr, *colidx;
            Data* values;
        } entry;

        struct {
            unsigned int rowptr, colidx, values;
        } offset;

    } pool;

    unsigned int workspace_nbyte;
    unsigned int dump_nbyte;

    struct {
        unsigned int rowptr, colidx, values, total;
    } bytelen;

    unsigned int m{0};
    unsigned int dummy_nnz{0};
    unsigned int nnz{0};

    /********************************************************************************
     * compression use
     ********************************************************************************/
    OutlierHandler(unsigned int _len);

    /**
     * @brief set up memory pool
     *
     * @param _pool
     * @param try_nnz
     */
    OutlierHandler& configure(uint8_t* _pool);

    // TODO handle nnz == 0 otherwise
    unsigned int query_csr_bytelen() const
    {
        return sizeof(int) * (m + 1)  // rowptr
               + sizeof(int) * nnz    // colidx
               + sizeof(Data) * nnz;  // values
    }

    /**
     * @brief use when the real nnz is known
     *
     * @param nnz
     */
    OutlierHandler& configure_with_nnz(int nnz);

    OutlierHandler& gather_CUDA10(float* in, unsigned int& dump_nbyte, float& ms_timer);

    OutlierHandler& archive(uint8_t* archive, int& nnz);

    /********************************************************************************
     * decompression use
     ********************************************************************************/
    OutlierHandler(unsigned int _len, unsigned int _nnz);

    OutlierHandler& extract(uint8_t* _pool);

    OutlierHandler& scatter_CUDA10(float*, float&);
};

#endif