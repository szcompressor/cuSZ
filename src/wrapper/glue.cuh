/**
 * @file glue.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-03-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef WRAPPER_GLUE_CUH
#define WRAPPER_GLUE_CUH

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime.h>

#include "../../include/reducer.hh"
#include "../utils/cuda_err.cuh"
#include "../utils/timer.hh"

// when using nvcc, functors must be defined outside a (__host__) function
template <typename E>
struct cleanup : public thrust::unary_function<E, E> {
    int radius;
    cleanup(int radius) : radius(radius) {}
    __host__ __device__ E operator()(const E e) const { return e; }
};

template <typename E, typename Policy, typename IDX = int, bool SHIFT = true>
void split_by_radius(
    E*           in_errctrl,
    size_t       in_len,
    int const    radius,
    IDX*         out_idx,
    E*           out_val,
    int&         out_nnz,
    cudaStream_t stream = nullptr,
    Policy       policy = thrust::device)
{
    using thrust::placeholders::_1;

    thrust::cuda::par.on(stream);
    thrust::counting_iterator<IDX> zero(0);

    // find out the indices
    out_nnz = thrust::copy_if(policy, zero, zero + in_len, in_errctrl, out_idx, _1 >= 2 * radius or _1 <= 0) - out_idx;

    // fetch corresponding values
    thrust::copy(
        policy, thrust::make_permutation_iterator(in_errctrl, out_idx),
        thrust::make_permutation_iterator(in_errctrl + out_nnz, out_idx + out_nnz), out_val);

    // clear up
    cleanup<E> functor(radius);
    thrust::transform(
        policy,                                                                      //
        thrust::make_permutation_iterator(in_errctrl, out_idx),                      //
        thrust::make_permutation_iterator(in_errctrl + out_nnz, out_idx + out_nnz),  //
        thrust::make_permutation_iterator(in_errctrl, out_idx),                      //
        functor);
}

template <typename E, typename Policy, typename IDX = int>
void split_by_binary_twopass(
    E*           in_errctrl,
    size_t       in_len,
    int const    radius,
    IDX*         out_idx,
    E*           out_val,
    int&         out_nnz,
    cudaStream_t stream = nullptr,
    Policy       policy = thrust::device)
{
    using thrust::placeholders::_1;

    thrust::cuda::par.on(stream);
    thrust::counting_iterator<IDX> zero(0);

    // find out the indices
    out_nnz = thrust::copy_if(policy, zero, zero + in_len, in_errctrl, out_idx, _1 != radius) - out_idx;

    // fetch corresponding values
    thrust::copy(
        policy, thrust::make_permutation_iterator(in_errctrl, out_idx),
        thrust::make_permutation_iterator(in_errctrl + out_nnz, out_idx + out_nnz), out_val);
}

// when using nvcc, functors must be defined outside a (__host__) function
template <typename Tuple>
struct is_outlier {
    int radius;
    is_outlier(int radius) : radius(radius) {}
    __host__ __device__ bool operator()(const Tuple t) const { return thrust::get<1>(t) != radius; }
};

template <typename E, typename Policy, typename IDX = int>
void split_by_binary_onepass(
    E*           in_errctrl,
    size_t       in_len,
    int const    radius,
    IDX*         out_idx,
    E*           out_val,
    int&         out_nnz,
    cudaStream_t stream = nullptr,
    Policy       policy = thrust::device)
{
    thrust::cuda::par.on(stream);
    using Tuple = thrust::tuple<IDX, E>;
    thrust::counting_iterator<IDX> zero(0);

    auto in      = thrust::make_zip_iterator(thrust::make_tuple(zero, in_errctrl));
    auto in_last = thrust::make_zip_iterator(thrust::make_tuple(zero + in_len, in_errctrl + in_len));
    auto out     = thrust::make_zip_iterator(thrust::make_tuple(out_idx, out_val));

    is_outlier<Tuple> functor(radius);
    out_nnz = thrust::copy_if(policy, in, in_last, out, functor) - out;
}

#define SPGS_DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};
#define SPGS_ALLOCDEV(VAR, SYM)                            \
    CHECK_CUDA(cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM])); \
    CHECK_CUDA(cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]));
#define SPGS_D2DCPY(VAR, FIELD)                                                                        \
    {                                                                                                  \
        auto dst = d_spfmt + header.entry[HEADER::FIELD];                                              \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

namespace cusz {

template <typename T = float>
class CompatibleSPGS : public VirtualGatherScatter {
   public:
    using Origin    = T;
    using BYTE      = uint8_t;
    using MetadataT = int;

   private:
    SPGS_DEFINE_ARRAY(spfmt, BYTE);
    SPGS_DEFINE_ARRAY(idx, int);
    SPGS_DEFINE_ARRAY(val, T);

    float milliseconds{0.0};

   public:
    struct header_t {
        static const int HEADER = 0;
        static const int IDX    = 1;
        static const int VAL    = 2;
        static const int END    = 3;

        int       header_nbyte : 16;
        size_t    uncompressed_len;  // TODO unnecessary?
        int       nnz;
        MetadataT entry[END + 1];

        MetadataT subfile_size() const { return entry[END]; }
    };
    using HEADER = struct header_t;

   private:
    struct runtime_encode_helper {
        static const int SPFMT = 0;
        static const int IDX   = 1;
        static const int VAL   = 2;
        static const int END   = 3;

        uint32_t nbyte[END];
        int      nnz{0};
        // HEADER* ptr_header{nullptr};
    };
    using RTE = runtime_encode_helper;
    RTE rte;

   public:
    float      get_time_elapsed() const { return milliseconds; }
    MetadataT* expose_idx() const { return d_idx; }
    T*         expose_val() const { return d_val; }

    void init(size_t const in_uncompressed_len, int density_factor = 4, bool dbg_print = false)
    {
        auto max_compressed_bytes = [&]() { return in_uncompressed_len / density_factor * sizeof(T); };
        auto init_nnz             = [&]() { return in_uncompressed_len / density_factor; };

        memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);
        rte.nnz = init_nnz();

        rte.nbyte[RTE::SPFMT] = max_compressed_bytes();
        rte.nbyte[RTE::IDX]   = rte.nnz * sizeof(int);
        rte.nbyte[RTE::VAL]   = rte.nnz * sizeof(T);

        SPGS_ALLOCDEV(spfmt, SPFMT);
        SPGS_ALLOCDEV(idx, IDX);
        SPGS_ALLOCDEV(val, VAL);

        // if (dbg_print) debug();
    }

   private:
    /**
     * @brief Collect fragmented arrays.
     *
     * @param header (host variable)
     * @param in_uncompressed_len (host variable)
     * @param stream CUDA stream
     */
    void
    subfile_collect(HEADER& header, size_t in_uncompressed_len, cudaStream_t stream = nullptr, bool dbg_print = false)
    {
        header.header_nbyte     = sizeof(HEADER);
        header.uncompressed_len = in_uncompressed_len;
        header.nnz              = rte.nnz;

        // update (redundant here)
        rte.nbyte[RTE::IDX] = sizeof(int) * rte.nnz;
        rte.nbyte[RTE::VAL] = sizeof(T) * rte.nnz;

        MetadataT nbyte[HEADER::END];
        nbyte[HEADER::HEADER] = 128;
        nbyte[HEADER::IDX]    = rte.nbyte[RTE::IDX];
        nbyte[HEADER::VAL]    = rte.nbyte[RTE::VAL];

        header.entry[0] = 0;
        // *.END + 1; need to knwo the ending position
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

        auto debug_header_entry = [&]() {
            printf("\nCSR11::subfile_collect() debugging:\n");
            printf("%-*s:  %'10ld\n", 16, "final.nnz", rte.nnz);
            printf("  ENTRIES\n");

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)HEADER::VAR, 14, #VAR, header.entry[HEADER::VAR]);
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

        SPGS_D2DCPY(idx, IDX)
        SPGS_D2DCPY(val, VAL)

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
    }

   public:
    void gather_detail(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    radius,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        int          method    = 0,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        HEADER header;

        if (method == 0)
            split_by_radius(
                in_uncompressed, in_uncompressed_len, radius, d_idx, d_val, rte.nnz, stream, thrust::device);

        else {
            if (method == 1)
                split_by_binary_onepass(
                    in_uncompressed, in_uncompressed_len, radius, d_idx, d_val, rte.nnz, stream, thrust::device);
            else if (method == 2)
                split_by_binary_twopass(
                    in_uncompressed, in_uncompressed_len, radius, d_idx, d_val, rte.nnz, stream, thrust::device);
        }

        subfile_collect(header, in_uncompressed_len, stream, dbg_print);
        out_compressed     = d_spfmt;
        out_compressed_len = header.subfile_size();
    }

    void gather_splitbyradius(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    radius,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        gather_detail(
            in_uncompressed, in_uncompressed_len, radius, out_compressed, out_compressed_len, 0, stream, dbg_print);
    }

    void gather_bianry1pass(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    radius,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        gather_detail(
            in_uncompressed, in_uncompressed_len, radius, out_compressed, out_compressed_len, 1, stream, dbg_print);
    }

    void gather_bianry2pass(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    radius,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        gather_detail(
            in_uncompressed, in_uncompressed_len, radius, out_compressed, out_compressed_len, 2, stream, dbg_print);
    }

    void scatter(BYTE* in_compressed, T* out_decompressed, cudaStream_t stream = nullptr)
    {
        header_t header;
        CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header.entry[HEADER::SYM])
        auto d_idx = ACCESSOR(IDX, int);
        auto d_val = ACCESSOR(VAL, T);
#undef ACCESSOR
        auto nnz = header.nnz;

        thrust::cuda::par.on(stream);
        cuda_timer_t t;
        t.timer_start(stream);
        thrust::scatter(thrust::device, d_val, d_val + nnz, d_idx, out_decompressed);
        t.timer_end(stream);
        milliseconds = t.get_time_elapsed();
    }

    void clear_buffer()
    {
        cudaMemset(d_spfmt, 0x0, rte.nbyte[RTE::SPFMT]);
        cudaMemset(d_idx, 0x0, rte.nbyte[RTE::IDX]);
        cudaMemset(d_val, 0x0, rte.nbyte[RTE::VAL]);
    }
};

}  // namespace cusz

#endif
