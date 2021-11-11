/**
 * @file ex_spline3_common.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-06-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include "../src/common.hh"
#include "../src/kernel/spline3.cuh"
#include "../src/utils.hh"
#include "../src/wrapper.hh"

/*
template <typename T>
void c_gather_anchor_cpu(
    T*           data,
    unsigned int dimx,
    unsigned int dimy,
    unsigned int dimz,
    dim3         leap,
    T*           anchor,
    dim3         anchor_leap)
{
    for (auto iz = 0, z = 0; z < dimz; iz++, z += 8) {
        for (auto iy = 0, y = 0; y < dimy; iy++, y += 8) {
            for (auto ix = 0, x = 0; x < dimx; ix++, x += 8) {
                auto data_id      = x + y * leap.y + z * leap.z;
                auto anchor_id    = ix + iy * anchor_leap.y + iz * anchor_leap.z;
                anchor[anchor_id] = data[data_id];
            }
        }
    }
}

template <typename T, bool PRINT_FP = false, bool PADDING = true>
void print_block_from_CPU(T* data, int radius = 512)
{
    cout << "dimxpad: " << dimx_aligned << "\tdimypad: " << dimy_aligned << '\n';

    for (auto z = 0; z < (BLOCK + (int)PADDING); z++) {
        printf("\nprint from CPU, z=%d\n", z);
        printf("    ");
        for (auto i = 0; i < 33; i++) printf("%3d", i);
        printf("\n");

        for (auto y = 0; y < (BLOCK + (int)PADDING); y++) {
            printf("y=%d ", y);

            for (auto x = 0; x < (4 * BLOCK + (int)PADDING); x++) {  //
                auto gid = x + y * dimx_aligned + z * dimx_aligned * dimy_aligned;
                if CONSTEXPR (PRINT_FP) { printf("%.2e\t", data[gid]); }
                else {
                    auto c = (int)data[gid] - radius;
                    if (c == 0)
                        printf("%3c", '.');
                    else {
                        if (abs(c) >= 10)
                            printf("%3c", '*');
                        else
                            printf("%3d", c);
                    }
                }
            }
            printf("\n");
        }
    }
    printf("\nCPU print end\n\n\n");

    printf("print *sv: \"x y z val\"\n");

    for (auto z = 0; z < (BLOCK + (int)PADDING); z++) {
        for (auto y = 0; y < (BLOCK + (int)PADDING); y++) {
            for (auto x = 0; x < (4 * BLOCK + (int)PADDING); x++) {  //
                auto gid = x + y * dimx_aligned + z * dimx_aligned * dimy_aligned;
                auto c   = (int)data[gid] - radius;
                if (c != 0) printf("%d %d %d %d\n", x, y, z, c);
            }
        }
    }
    printf("\n");
}
*/

/**
 * @brief
 *
 * @tparam SP_FACTOR presumed outlier percentage, default to 1/10
 */
template <int SP_FACTOR = 10>
class TestSpline3Wrapped {
    using T    = float;    // single-precision original data
    using E    = float;    // error-control code is in float
    using P    = float;    // internal precision
    using BYTE = uint8_t;  // non-interpreted type; bytestream

    static const bool USE_UNFIED = false;  // disable unified memory

    unsigned int len;
    dim3         data_size;

    static const auto BLOCK  = 8;
    static const auto radius = 0;

    static const auto MODE       = cusz::DEV::TEST;
    static const auto BOTH       = cusz::LOC::HOST_DEVICE;
    static const auto EXEC_SPACE = cusz::LOC::DEVICE;
    static const auto ALT_SPACE  = cusz::LOC::HOST;

    double in_eb, eb, eb_r, ebx2, ebx2_r;
    double max_value, min_value, rng;

    Capsule<T, USE_UNFIED> data;   // may need .prescan() to determine the range
    Capsule<T, USE_UNFIED> xdata;  // may need .device2host()
    Capsule<T, USE_UNFIED> anchor;
    Capsule<E, USE_UNFIED> errctrl;

    Capsule<BYTE, USE_UNFIED> compress_dump;
    Capsule<BYTE, USE_UNFIED> sp_use2;

    int  m;       // nrow of reinterpreted matrix
    int* rowptr;  // outside allocated CSR-rowptr
    int* colidx;  // outside allocated CSR-colidx
    T*   values;  // outside allocated CSR-values

    bool use_outer_space = false;

    std::string fname;
    stat_t      stat;

    cusz::Spline3<T, E, P>* predictor;
    cusz::CSR11<E>*         spreducer_c;
    cusz::CSR11<E>*         spreducer_d;

    uint32_t sp_dump_nbyte;
    int      nnz{0};

   public:
    uint32_t get_data_len() { return data_size.x * data_size.y * data_size.z; }
    uint32_t get_exact_spdump_nbyte() { return sp_dump_nbyte; }
    uint32_t get_anchor_len() { return predictor->get_anchor_len(); }
    uint32_t get_nnz() { return nnz; }

   private:
    void init_space()
    {
        m = Reinterpret1DTo2D::get_square_size(predictor->get_quant_len());
        cudaMalloc(&rowptr, sizeof(int) * (m + 1));
        cudaMalloc(&colidx, sizeof(int) * (len / SP_FACTOR));
        cudaMalloc(&values, sizeof(T) * (len / SP_FACTOR));
    }

    void init_space(int*& outer_rowptr, int*& outer_colidx, T*& outer_values)
    {
        use_outer_space = true;

        m      = Reinterpret1DTo2D::get_square_size(predictor->get_quant_len());
        rowptr = outer_rowptr;
        colidx = outer_colidx;
        values = outer_values;
    }

    void init_print_dbg()
    {
        if (fname != "") std::cout << "opening " << fname << std::endl;
        std::cout << "input eb: " << in_eb << '\n';
        std::cout << "range: " << rng << '\n';
        std::cout << "r2r eb: " << eb << '\n';
        std::cout << "predictor.anchor_len() = " << predictor->get_anchor_len() << '\n';
        std::cout << "predictor.quant_len() = " << predictor->get_quant_len() << '\n';
    }

   public:
    void data_analysis(std::string _fname)
    {
        if (_fname != "")
            data.from_fs_to<ALT_SPACE>(_fname);
        else {
            throw std::runtime_error("need to specify data source");
        }

        analysis::verify_data<T>(&stat, xdata.get<ALT_SPACE>(), data.get<ALT_SPACE>(), len);
        analysis::print_data_quality_metrics<T>(&stat, 0, false);

        auto compressed_size = sp_dump_nbyte + predictor->get_anchor_len() * sizeof(T);
        auto original_size   = (data.get_len()) * sizeof(T);
        LOGGING(LOG_INFO, "compression ratio: ", 1.0 * original_size / compressed_size);
    }

    void data_analysis(T*& h_ext_data)
    {
        data.from_existing_on<ALT_SPACE>(h_ext_data);
        xdata.template alloc<ALT_SPACE>().device2host();

        analysis::verify_data<T>(&stat, xdata.get<ALT_SPACE>(), data.get<ALT_SPACE>(), len);
        analysis::print_data_quality_metrics<T>(&stat, 0, false);

        auto compressed_size = sp_dump_nbyte + predictor->get_anchor_len() * sizeof(T);
        auto original_size   = (data.get_len()) * sizeof(T);
        LOGGING(LOG_INFO, "compression ratio: ", 1.0 * original_size / compressed_size);

        xdata.template free<ALT_SPACE>();
    }

   public:
    TestSpline3Wrapped(
        T*     _data,
        dim3   _size,
        double _eb,
        int*   outer_rowptr = nullptr,
        int*   outer_colidx = nullptr,
        T*     outer_values = nullptr,
        bool   verbose      = false)
    {
        data_size = _size;
        len       = get_data_len();

        data  //
            .set_len(len)
            .from_existing_on<EXEC_SPACE>(_data);
        xdata.set_len(len);

        // set eb
        in_eb = _eb, eb = _eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

        predictor = new cusz::Spline3<T, E, P>(data_size, eb, radius);

        if (outer_rowptr and outer_colidx and outer_values)
            init_space(outer_rowptr, outer_colidx, outer_values);
        else
            init_space();

        spreducer_c = new cusz::CSR11<E>(predictor->get_quant_len(), rowptr, colidx, values);

        anchor  //
            .set_len(predictor->get_anchor_len())
            .alloc<EXEC_SPACE>();
        errctrl  //
            .set_len(predictor->get_quant_len())
            .alloc<EXEC_SPACE, cusz::ALIGNDATA::SQUARE_MATRIX>();

        if (verbose) init_print_dbg();
    }

    void iterative_stacking()
    {
        Capsule<T, USE_UNFIED> stack_img(xdata.get_len());
        stack_img.alloc<cusz::LOC::HOST_DEVICE>();

        predictor->construct(data.get<EXEC_SPACE>(), anchor.get<EXEC_SPACE>(), errctrl.get<EXEC_SPACE>());
        predictor->reconstruct(anchor.get<EXEC_SPACE>(), errctrl.get<EXEC_SPACE>(), xdata.get<EXEC_SPACE>());

        thrust::transform(
            xdata.dptr, xdata.dptr + xdata.get_len(), stack_img.dptr, stack_img.dptr, thrust::plus<float>());
    }

    void compress2()
    {
        predictor->construct(data.get<EXEC_SPACE>(), anchor.get<EXEC_SPACE>(), errctrl.get<EXEC_SPACE>());
        spreducer_c->gather(errctrl.get<EXEC_SPACE>(), sp_dump_nbyte, nnz);

        errctrl.memset();
    }

    void export_after_compress2(uint8_t* d_spdump, T* d_anchordump)
    {
        // a memory copy
        spreducer_c->template consolidate<EXEC_SPACE, EXEC_SPACE>(d_spdump);
        // a memory copy
        cudaMemcpy(d_anchordump, anchor.get<EXEC_SPACE>(), anchor.nbyte(), cudaMemcpyDeviceToDevice);
    }

    void decompress2(T* d_xdata, uint8_t* d_spdump, int _nnz, T* d_anchordump)
    {
        xdata.template from_existing_on<EXEC_SPACE>(d_xdata);
        nnz         = _nnz;
        spreducer_d = new cusz::CSR11<E>(predictor->get_quant_len(), nnz);
        LOGGING(LOG_INFO, "nnz:", nnz);

        spreducer_d->scatter(d_spdump, errctrl.get<EXEC_SPACE>());
        predictor->reconstruct(d_anchordump, errctrl.get<EXEC_SPACE>(), xdata.get<EXEC_SPACE>());
    }

    ~TestSpline3Wrapped()
    {
        errctrl.free<EXEC_SPACE>();
        anchor.free<EXEC_SPACE>();
        data.free<EXEC_SPACE>();
        xdata.free<EXEC_SPACE>();

        if (not use_outer_space) {
            cudaFree(rowptr);
            cudaFree(colidx);
            cudaFree(values);
        }
    }
};
