/**
 * @file extrap_lorenzo.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_EXTRAP_LORENZO_CUH
#define CUSZ_WRAPPER_EXTRAP_LORENZO_CUH

#include "../../include/predictor.hh"

template <typename T = float, typename E = float, typename FP = float>
void tbd_lorenzo_dryrun(T* data, dim3 size3, int ndim, FP eb);

template <typename T = float, typename E = float, typename FP = float, bool DELAY_POSTQUANT = false>
void compress_lorenzo_construct(T* data, E* quant, dim3 size3, int ndim, FP eb, int radius, float& ms);

template <typename T = float, typename E = float, typename FP = float, bool DELAY_POSTQUANT = false>
void decompress_lorenzo_reconstruct(T* data, E* quant, dim3 size3, int ndim, FP eb, int radius, float& ms);

namespace cusz {

template <typename T, typename E, typename FP>
class PredictorLorenzo : public PredictorAbstraction<T, E> {
   public:
    using Precision = FP;

   private:
    int    radius;
    double eb;
    FP     ebx2_r;
    FP     ebx2;

    dim3     size;  // size.x, size.y, size.z
    dim3     leap;  // leap.y, leap.z
    int      ndim;
    uint32_t len_data;
    uint32_t len_quant;  // may differ from `len_data`
    bool     delay_postquant;

    float time_elapsed;

    struct {
        bool count_nnz;
        // bool blockwide_gather; // future use
    } on_off;

    template <bool DELAY_POSTQUANT>
    void construct_proxy(T* in_data, T* out_anchor, E* out_errctrl);

    template <bool DELAY_POSTQUANT>
    void reconstruct_proxy(T* in_anchor, E* in_errctrl, T* out_xdata);

   public:
    // context free
    PredictorLorenzo(dim3 xyz, double eb, int radius, bool delay_postquant);

    // helper
    uint32_t get_quant_len() const { return len_quant; }
    uint32_t get_anchor_len() const { return 0; }
    float    get_time_elapsed() const { return time_elapsed; }
    uint32_t get_workspace_nbyte() const { return 0; };

    // methods
    void dryrun(T* in_out);

    void construct(T* in_data, T* out_anchor, E* out_errctrl);

    void reconstruct(T* in_anchor, E* in_errctrl, T* out_xdata);
};

}  // namespace cusz

#endif