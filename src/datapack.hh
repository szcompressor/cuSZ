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

template <typename T>
struct PartialData {
    using type = T;
    T*           dptr;
    T*           hptr;
    unsigned int len;
    unsigned int nbyte() const { return len * sizeof(T); };

    PartialData(unsigned int _len = 0) { len = _len; }
    void memset(unsigned char init = 0x0u) { cudaMemset(dptr, init, nbyte()); }
    void h2d() { cudaMemcpy(dptr, hptr, nbyte(), cudaMemcpyHostToDevice); }
    void d2h() { cudaMemcpy(hptr, dptr, nbyte(), cudaMemcpyDeviceToHost); }
};

#endif