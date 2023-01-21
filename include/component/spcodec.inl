/**
 * @file spcodec_vec.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CF358238_3946_4FFC_B5E6_45C12F0C0B44
#define CF358238_3946_4FFC_B5E6_45C12F0C0B44

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "../common.hh"
#include "../kernel/spv_gpu.hh"
#include "utils/cuda_err.cuh"

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

#define SPVEC_ALLOCDEV(VAR, SYM)                           \
    CHECK_CUDA(cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM])); \
    CHECK_CUDA(cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]));

#define SPVEC_FREEDEV(VAR)             \
    if (d_##VAR) {                     \
        CHECK_CUDA(cudaFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define SPVEC_D2DCPY(VAR, FIELD)                                                                       \
    {                                                                                                  \
        auto dst = d_spfmt + header.entry[Header::FIELD];                                              \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[Header::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

namespace cusz {

/*******************************************************************************
 * sparsity-aware coder/decoder, vector
 *******************************************************************************/

template <typename T, typename M = uint32_t>
class SpcodecVec {
   public:
    using Origin    = T;
    using BYTE      = uint8_t;
    using MetadataT = M;

    struct Header {
        static const int HEADER = 0;
        static const int IDX    = 1;
        static const int VAL    = 2;
        static const int END    = 3;

        int       header_nbyte : 16;
        size_t    uncompressed_len;
        int       nnz;
        MetadataT entry[END + 1];

        MetadataT subfile_size() const { return entry[END]; }
    };

    struct runtime_encode_helper {
        static const int SPFMT = 0;
        static const int IDX   = 1;
        static const int VAL   = 2;
        static const int END   = 3;

        uint32_t nbyte[END];
        int      nnz{0};
    };

   private:
    DEFINE_ARRAY(spfmt, BYTE);
    DEFINE_ARRAY(idx, M);
    DEFINE_ARRAY(val, T);

    using RTE = runtime_encode_helper;

    float milliseconds{0.0};

    RTE rte;

   public:
    ~SpcodecVec()
    {
        SPVEC_FREEDEV(spfmt);
        SPVEC_FREEDEV(idx);
        SPVEC_FREEDEV(val);
    }                                          // dtor
    SpcodecVec() {}                            // ctor
    SpcodecVec(const SpcodecVec&);             // copy ctor
    SpcodecVec& operator=(const SpcodecVec&);  // copy assign
    SpcodecVec(SpcodecVec&&);                  // move ctor
    SpcodecVec& operator=(SpcodecVec&&);       // move assign

    void init(size_t const len, int density_factor = 4, bool dbg_print = false)
    {
        auto max_bytes = [&]() { return len / density_factor * sizeof(T); };
        auto init_nnz  = [&]() { return len / density_factor; };

        memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);
        rte.nnz = init_nnz();

        rte.nbyte[RTE::SPFMT] = max_bytes();
        rte.nbyte[RTE::IDX]   = rte.nnz * sizeof(int);
        rte.nbyte[RTE::VAL]   = rte.nnz * sizeof(T);

        SPVEC_ALLOCDEV(spfmt, SPFMT);
        SPVEC_ALLOCDEV(idx, IDX);
        SPVEC_ALLOCDEV(val, VAL);

        // if (dbg_print) debug();
    }

    void encode(
        T*           in,
        size_t const in_len,
        BYTE*&       out,
        size_t&      out_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        Header header;

        accsz::spv_gather<T, M>(in, in_len, this->d_val, this->d_idx, &rte.nnz, &milliseconds, stream);

        subfile_collect(header, in_len, stream, dbg_print);
        out     = d_spfmt;
        out_len = header.subfile_size();
    }

    void decode(BYTE* coded, T* decoded, cudaStream_t stream = nullptr)
    {
        Header header;
        CHECK_CUDA(cudaMemcpyAsync(&header, coded, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(coded + header.entry[Header::SYM])
        auto d_idx = ACCESSOR(IDX, uint32_t);
        auto d_val = ACCESSOR(VAL, T);
#undef ACCESSOR

        accsz::spv_scatter<T, M>(d_val, d_idx, header.nnz, decoded, &milliseconds, stream);
    }

    void clear_buffer()
    {
        cudaMemset(d_spfmt, 0x0, rte.nbyte[RTE::SPFMT]);
        cudaMemset(d_idx, 0x0, rte.nbyte[RTE::IDX]);
        cudaMemset(d_val, 0x0, rte.nbyte[RTE::VAL]);
    }

    float get_time_elapsed() const { return milliseconds; }

    void subfile_collect(Header& header, size_t len, cudaStream_t stream, bool dbg_print)
    {
        header.header_nbyte     = sizeof(Header);
        header.uncompressed_len = len;
        header.nnz              = rte.nnz;

        // update (redundant here)
        rte.nbyte[RTE::IDX] = sizeof(int) * rte.nnz;
        rte.nbyte[RTE::VAL] = sizeof(T) * rte.nnz;

        MetadataT nbyte[Header::END];
        nbyte[Header::HEADER] = 128;
        nbyte[Header::IDX]    = rte.nbyte[RTE::IDX];
        nbyte[Header::VAL]    = rte.nbyte[RTE::VAL];

        header.entry[0] = 0;
        // *.END + 1; need to knwo the ending position
        for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
        for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

        auto debug_header_entry = [&]() {
            printf("\nCSR11::subfile_collect() debugging:\n");
            printf("%-*s:  %'10ld\n", 16, "final.nnz", rte.nnz);
            printf("  ENTRIES\n");

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)Header::VAR, 14, #VAR, header.entry[Header::VAR]);
            PRINT_ENTRY(HEADER);
            PRINT_ENTRY(IDX);
            PRINT_ENTRY(VAL);
            PRINT_ENTRY(END);
            printf("\n");
#undef PRINT_ENTRY
        };
        if (dbg_print) debug_header_entry();

        CHECK_CUDA(cudaMemcpyAsync(d_spfmt, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        SPVEC_D2DCPY(idx, IDX)
        SPVEC_D2DCPY(val, VAL)

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
    }
};

}  // namespace cusz

#undef DEFINE_ARRAY
#undef SPVEC_ALLOCDEV
#undef SPVEC_FREEDEV
#undef SPVEC_D2DCPY

#endif /* CF358238_3946_4FFC_B5E6_45C12F0C0B44 */
