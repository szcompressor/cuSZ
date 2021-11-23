/**
 * @file sp_path.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-29
 * meged file created on 2021-06-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_SP_PATH_CUH
#define CUSZ_SP_PATH_CUH

#include "base_cusz.cuh"
#include "binding.hh"
#include "header.hh"
#include "wrapper.hh"
// #include "wrapper/interp_spline3.cuh"

template <class BINDING>
class SpPathCompressorOld : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor = typename BINDING::PREDICTOR;
    using SpReducer = typename BINDING::SPREDUCER;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;

   private:
    // --------------------
    // not in base class
    // --------------------
    Capsule<BYTE> sp_use;

    Predictor* predictor;
    SpReducer* spreducer;

    size_t   m, mxm;
    uint32_t sp_dump_nbyte;

    static const auto EXEC_SPACE     = cusz::LOC::DEVICE;
    static const auto FALLBACK_SPACE = cusz::LOC::HOST;
    static const auto BOTH           = cusz::LOC::HOST_DEVICE;

   private:
   public:
    SpPathCompressorOld(cuszCTX* _ctx, Capsule<T>* _in_data);
    SpPathCompressorOld(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);
    ~SpPathCompressorOld();

    SpPathCompressorOld& compress();

    template <cusz::LOC SRC, cusz::LOC DST>
    SpPathCompressorOld& consolidate(BYTE** dump);

    SpPathCompressorOld& decompress(Capsule<T>* out_xdata);
    SpPathCompressorOld& backmatter(Capsule<T>* out_xdata);
};

struct SparsityAwarePathOld {
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<4, true>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using DefaultBinding = PredictorReducerBinding<  //
        cusz::Spline3<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using DefaultCompressor = class SpPathCompressorOld<DefaultBinding>;

    using FallbackBinding = PredictorReducerBinding<  //
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using FallbackCompressor = class SpPathCompressorOld<FallbackBinding>;
};

template <typename T, cusz::LOC LOC = cusz::LOC::DEVICE>
struct AdHocSpArchive {
   private:
    using BYTE = uint8_t;
    Capsule<BYTE> space;
    cuszHEADER*   header;

    unsigned int m;

    struct {
        unsigned int rowptr;
        unsigned int colidx;
        unsigned int values;
    } nbyte;

   public:
    cuszHEADER* get_header() { return header; }
    int         get_nnz() { return header->nnz_outlier; }
    T*          get_anchor() { return reinterpret_cast<T*>(space.template get<LOC>()); }
    BYTE*       get_spdump() { return space.template get<LOC>() + header->nbyte.anchor; }

   public:
    AdHocSpArchive(cuszHEADER* _header, unsigned int len, unsigned int assumed_sp_ratio = 10 /* 5 */)
    {
        header = _header;
        space.set_len(len / assumed_sp_ratio * 3).template alloc<LOC>();
        m = Reinterpret1DTo2D::get_square_size(len);
    }

    AdHocSpArchive& set_nnz(int nnz)
    {
        header->nnz_outlier = nnz;
        nbyte.rowptr        = sizeof(int) * (m + 1);
        nbyte.colidx        = sizeof(int) * nnz;
        nbyte.colidx        = sizeof(T) * nnz;
        return *this;
    }

    AdHocSpArchive& set_anchor_len(unsigned int anchor_len)
    {
        header->nbyte.anchor = anchor_len * sizeof(T);
        return *this;
    }
};

typedef struct AdHocSpArchive<float, cusz::LOC::DEVICE> SpArchive;

template <class BINDING, int SP_FACTOR = 10>
class SpPathCompressor : public BaseCompressor<typename BINDING::PREDICTOR> {
    using Predictor = typename BINDING::PREDICTOR;
    using SpReducer = typename BINDING::SPREDUCER;

    using T    = typename Predictor::Origin;   // wrong in type inference
    using E    = typename Predictor::ErrCtrl;  // wrong in type inference
    using BYTE = uint8_t;                      // non-interpreted type; bytestream

    // static const bool USE_UNFIED = false;  // disable unified memory

    unsigned int len;
    dim3         data_size;

    static const auto BLOCK  = 8;
    static const auto radius = 0;

    static const auto MODE   = cusz::DEV::TEST;
    static const auto BOTH   = cusz::LOC::HOST_DEVICE;
    static const auto DEVICE = cusz::LOC::DEVICE;
    static const auto HOST   = cusz::LOC::HOST;

    double in_eb, eb, eb_r, ebx2, ebx2_r;
    double max_value, min_value, rng;

    Capsule<T> data;   // may need .prescan() to determine the range
    Capsule<T> xdata;  // may need .device2host()
    Capsule<T> anchor;
    Capsule<E> errctrl;

    Capsule<BYTE> compress_dump;
    Capsule<BYTE> sp_use2;

    int  m;       // nrow of reinterpreted matrix
    int* rowptr;  // outside allocated CSR-rowptr
    int* colidx;  // outside allocated CSR-colidx
    T*   values;  // outside allocated CSR-values

    bool use_outer_space = false;

    std::string fname;
    stat_t      stat;

    Predictor* predictor;
    SpReducer* spreducer;

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
            data.from_fs_to<HOST>(_fname);
        else {
            throw std::runtime_error("need to specify data source");
        }

        analysis::verify_data<T>(&stat, xdata.template get<HOST>(), data.template get<HOST>(), len);
        analysis::print_data_quality_metrics<T>(&stat, 0, false);

        auto compressed_size = sp_dump_nbyte + predictor->get_anchor_len() * sizeof(T);
        auto original_size   = (data.get_len()) * sizeof(T);
        LOGGING(LOG_INFO, "compression ratio: ", 1.0 * original_size / compressed_size);
    }

    void data_analysis(T*& h_ext_data)
    {
        data.template from_existing_on<HOST>(h_ext_data);
        xdata.template alloc<HOST>().device2host();

        analysis::verify_data<T>(&stat, xdata.template get<HOST>(), data.template get<HOST>(), len);
        analysis::print_data_quality_metrics<T>(&stat, 0, false);

        auto compressed_size = sp_dump_nbyte + predictor->get_anchor_len() * sizeof(T);
        auto original_size   = (data.get_len()) * sizeof(T);
        LOGGING(LOG_INFO, "compression ratio: ", 1.0 * original_size / compressed_size);

        xdata.template free<HOST>();
    }

   public:
    SpPathCompressor(
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

        data.set_len(len).template from_existing_on<DEVICE>(_data);
        xdata.set_len(len);

        // set eb
        in_eb = _eb, eb = _eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

        predictor = new Predictor(data_size, eb, radius);

        if (outer_rowptr and outer_colidx and outer_values)
            init_space(outer_rowptr, outer_colidx, outer_values);
        else
            init_space();

        spreducer = new SpReducer(predictor->get_quant_len());
        spreducer->compress_set_space(rowptr, colidx, values);

        anchor  //
            .set_len(predictor->get_anchor_len())
            .template alloc<DEVICE>();
        errctrl  //
            .set_len(predictor->get_quant_len())
            .template alloc<DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();

        if (verbose) init_print_dbg();
    }

    void iterative_stacking()
    {
        Capsule<T> stack_img(xdata.get_len());
        stack_img.template alloc<cusz::LOC::HOST_DEVICE>();

        predictor->construct(
            data.template get<DEVICE>(), anchor.template get<DEVICE>(), errctrl.template get<DEVICE>());
        predictor->reconstruct(
            anchor.template get<DEVICE>(), errctrl.template get<DEVICE>(), xdata.template get<DEVICE>());

        thrust::transform(
            xdata.dptr, xdata.dptr + xdata.get_len(), stack_img.dptr, stack_img.dptr, thrust::plus<float>());
    }

    void compress()
    {
        predictor->construct(
            data.template get<DEVICE>(), anchor.template get<DEVICE>(), errctrl.template get<DEVICE>());
        spreducer->gather(errctrl.template get<DEVICE>(), sp_dump_nbyte, nnz);

        errctrl.memset();
    }

    // TODO consolidate
    void export_after_compress(uint8_t* d_spdump, T* d_anchordump)
    {
        // a memory copy
        spreducer->template consolidate<DEVICE, DEVICE>(d_spdump);
        // a memory copy
        cudaMemcpy(d_anchordump, anchor.template get<DEVICE>(), anchor.nbyte(), cudaMemcpyDeviceToDevice);
    }

    void decompress(T* d_xdata, uint8_t* d_spdump, int _nnz, T* d_anchordump)
    {
        xdata.template from_existing_on<DEVICE>(d_xdata);
        spreducer->decompress_set_nnz(nnz);
        LOGGING(LOG_INFO, "nnz:", _nnz);

        spreducer->scatter(d_spdump, errctrl.template get<DEVICE>());
        predictor->reconstruct(d_anchordump, errctrl.template get<DEVICE>(), xdata.template get<DEVICE>());
    }

    ~SpPathCompressor()
    {
        errctrl.template free<DEVICE>();
        anchor.template free<DEVICE>();

        if (not use_outer_space) {
            cudaFree(rowptr);
            cudaFree(colidx);
            cudaFree(values);
        }
    }
};

struct SparsityAwarePath {
   private:
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<4, true>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

   public:
    using DefaultBinding = PredictorReducerBinding<  //
        cusz::Spline3<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using DefaultCompressor = class SpPathCompressor<DefaultBinding, 10>;

    using FallbackBinding = PredictorReducerBinding<  //
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using FallbackCompressor = class SpPathCompressor<FallbackBinding, 10>;
};

#endif
