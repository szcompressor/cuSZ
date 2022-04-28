/**
 * @file predictors.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "component/predictors.hh"

namespace cusz {

/*******************************************************************************
 * predictor Lorenzo
 *******************************************************************************/

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>::~PredictorLorenzo()
{
    pimpl.reset();
}

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>::PredictorLorenzo() : pimpl{std::make_unique<impl>()}
{
}

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>::PredictorLorenzo(const PredictorLorenzo<T, E, FP>& old) : pimpl(new impl(*old.pimpl))
{
    // TODO allocation/deep copy
}

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>& PredictorLorenzo<T, E, FP>::operator=(const PredictorLorenzo<T, E, FP>& old)
{
    *pimpl = *old.pimpl;
    // TODO allocation/deep copy
    return *this;
}

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>::PredictorLorenzo(PredictorLorenzo<T, E, FP>&&) = default;

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>& PredictorLorenzo<T, E, FP>::operator=(PredictorLorenzo<T, E, FP>&&) = default;

//------------------------------------------------------------------------------

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::init(size_t x, size_t y, size_t z, bool dbg_print)
{
    pimpl->init(x, y, z, dbg_print);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::init(dim3 xyz, bool dbg_print)
{
    pimpl->init(xyz, dbg_print);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::construct(
    dim3 const len3,
    T* __restrict__ in_data,
    T*& out_anchor,
    E*& out_errctrl,
    T*& __restrict__ out_outlier,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->construct(len3, in_data, out_anchor, out_errctrl, out_outlier, eb, radius, stream);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::reconstruct(
    dim3 len3,
    T* __restrict__ in_outlier,
    T* in_anchor,
    E* in_errctrl,
    T*& __restrict__ out_xdata,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->reconstruct(len3, in_outlier, in_anchor, in_errctrl, out_xdata, eb, radius, stream);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::construct(
    dim3 const   len3,
    T*           in_data__out_outlier,
    T*&          out_anchor,
    E*&          out_errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->construct(len3, in_data__out_outlier, out_anchor, out_errctrl, eb, radius, stream);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::reconstruct(
    dim3         len3,
    T*&          in_outlier__out_xdata,
    T*           in_anchor,
    E*           in_errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->reconstruct(len3, in_outlier__out_xdata, in_anchor, in_errctrl, eb, radius, stream);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::clear_buffer()
{
    pimpl->clear_buffer();
}

// getter
template <typename T, typename E, typename FP>
float PredictorLorenzo<T, E, FP>::get_time_elapsed() const
{
    return pimpl->get_time_elapsed();
}

template <typename T, typename E, typename FP>
size_t PredictorLorenzo<T, E, FP>::get_alloclen_data() const
{
    return pimpl->get_alloclen_data();
}

template <typename T, typename E, typename FP>
size_t PredictorLorenzo<T, E, FP>::get_alloclen_quant() const
{
    return pimpl->get_alloclen_quant();
}

template <typename T, typename E, typename FP>
size_t PredictorLorenzo<T, E, FP>::get_len_data() const
{
    return pimpl->get_len_data();
}

template <typename T, typename E, typename FP>
size_t PredictorLorenzo<T, E, FP>::get_len_quant() const
{
    return pimpl->get_len_quant();
}

template <typename T, typename E, typename FP>
size_t PredictorLorenzo<T, E, FP>::get_len_anchor() const
{
    return pimpl->get_len_anchor();
}

template <typename T, typename E, typename FP>
E* PredictorLorenzo<T, E, FP>::expose_quant() const
{
    return pimpl->expose_quant();
}

template <typename T, typename E, typename FP>
E* PredictorLorenzo<T, E, FP>::expose_errctrl() const
{
    return pimpl->expose_errctrl();
}

template <typename T, typename E, typename FP>
T* PredictorLorenzo<T, E, FP>::expose_anchor() const
{
    return pimpl->expose_anchor();
}

template <typename T, typename E, typename FP>
T* PredictorLorenzo<T, E, FP>::expose_outlier() const
{
    return pimpl->expose_outlier();
}

/*******************************************************************************
 * predictor Spline3
 *******************************************************************************/

template <typename T, typename E, typename FP>
PredictorSpline3<T, E, FP>::~PredictorSpline3()
{
    pimpl.reset();
}

template <typename T, typename E, typename FP>
PredictorSpline3<T, E, FP>::PredictorSpline3() : pimpl{std::make_unique<impl>()}
{
}

template <typename T, typename E, typename FP>
PredictorSpline3<T, E, FP>::PredictorSpline3(const PredictorSpline3<T, E, FP>& old) : pimpl(new impl(*old.pimpl))
{
    // TODO allocation/deep copy
}

template <typename T, typename E, typename FP>
PredictorSpline3<T, E, FP>& PredictorSpline3<T, E, FP>::operator=(const PredictorSpline3<T, E, FP>& old)
{
    *pimpl = *old.pimpl;
    // TODO allocation/deep copy
    return *this;
}

template <typename T, typename E, typename FP>
PredictorSpline3<T, E, FP>::PredictorSpline3(PredictorSpline3<T, E, FP>&& predictor) = default;

template <typename T, typename E, typename FP>
PredictorSpline3<T, E, FP>& PredictorSpline3<T, E, FP>::operator=(PredictorSpline3<T, E, FP>&&) = default;

//------------------------------------------------------------------------------

template <typename T, typename E, typename FP>
void PredictorSpline3<T, E, FP>::init(size_t x, size_t y, size_t z, bool dbg_print)
{
    pimpl->init(x, y, z, dbg_print);
}

template <typename T, typename E, typename FP>
void PredictorSpline3<T, E, FP>::init(dim3 xyz, bool dbg_print)
{
    pimpl->init(xyz, dbg_print);
}

template <typename T, typename E, typename FP>
void PredictorSpline3<T, E, FP>::construct(
    dim3 const   len3,
    T*           in_data__out_outlier,
    T*&          out_anchor,
    E*&          out_errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->construct(len3, in_data__out_outlier, out_anchor, out_errctrl, eb, radius, stream);
}

template <typename T, typename E, typename FP>
void PredictorSpline3<T, E, FP>::reconstruct(
    dim3         len3,
    T*&          in_outlier__out_xdata,
    T*           in_anchor,
    E*           in_errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->reconstruct(len3, in_outlier__out_xdata, in_anchor, in_errctrl, eb, radius, stream);
}

template <typename T, typename E, typename FP>
void PredictorSpline3<T, E, FP>::clear_buffer()
{
    pimpl->clear_buffer();
}

// getter
template <typename T, typename E, typename FP>
float PredictorSpline3<T, E, FP>::get_time_elapsed() const
{
    return pimpl->get_time_elapsed();
}

template <typename T, typename E, typename FP>
size_t PredictorSpline3<T, E, FP>::get_alloclen_data() const
{
    return pimpl->get_alloclen_data();
}

template <typename T, typename E, typename FP>
size_t PredictorSpline3<T, E, FP>::get_alloclen_quant() const
{
    return pimpl->get_alloclen_quant();
}

template <typename T, typename E, typename FP>
size_t PredictorSpline3<T, E, FP>::get_len_data() const
{
    return pimpl->get_len_data();
}

template <typename T, typename E, typename FP>
size_t PredictorSpline3<T, E, FP>::get_len_quant() const
{
    return pimpl->get_len_quant();
}

template <typename T, typename E, typename FP>
size_t PredictorSpline3<T, E, FP>::get_len_anchor() const
{
    return pimpl->get_len_anchor();
}

template <typename T, typename E, typename FP>
E* PredictorSpline3<T, E, FP>::expose_quant() const
{
    return pimpl->expose_quant();
}

template <typename T, typename E, typename FP>
E* PredictorSpline3<T, E, FP>::expose_errctrl() const
{
    return pimpl->expose_errctrl();
}

template <typename T, typename E, typename FP>
T* PredictorSpline3<T, E, FP>::expose_anchor() const
{
    return pimpl->expose_anchor();
}

template <typename T, typename E, typename FP>
T* PredictorSpline3<T, E, FP>::expose_outlier() const
{
    return pimpl->expose_outlier();
}

}  // namespace cusz

template struct cusz::PredictorLorenzo<float, uint16_t, float>;
template struct cusz::PredictorLorenzo<float, uint32_t, float>;
template struct cusz::PredictorLorenzo<float, float, float>;

template struct cusz::PredictorSpline3<float, uint32_t, float>;
template struct cusz::PredictorSpline3<float, float, float>;
