/**
 * @file csr11.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "csr11.cuh"

#define CSR11_TYPE cusz::CSR11<float>

template class CSR11_TYPE;

template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(uint8_t*);
template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::HOST, cusz::LOC::DEVICE>(uint8_t*);
template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::DEVICE, cusz::LOC::HOST>(uint8_t*);
template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::DEVICE, cusz::LOC::DEVICE>(uint8_t*);

#undef CSR11_TYPE
