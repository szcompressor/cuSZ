/**
 * @file prediction.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#define THE_TYPE template <typename T, typename E, typename FP>
#define PREDICTION PredictionUnified<T, E, FP>

#include "component/prediction.hh"

namespace cusz {

/*******************************************************************************
 * predictor Lorenzo
 *******************************************************************************/

THE_TYPE
PREDICTION::~PredictionUnified() { pimpl.reset(); }

THE_TYPE
PREDICTION::PredictionUnified() : pimpl{std::make_unique<impl>()} {}

THE_TYPE
PREDICTION::PredictionUnified(const PREDICTION& old) : pimpl{std::make_unique<impl>(*old.pimpl)}
{
    // TODO allocation/deep copy
}

THE_TYPE
PREDICTION& PREDICTION::operator=(const PREDICTION& old)
{
    *pimpl = *old.pimpl;
    // TODO allocation/deep copy
    return *this;
}

THE_TYPE
PREDICTION::PredictionUnified(PREDICTION&&) = default;

THE_TYPE
PREDICTION& PREDICTION::operator=(PREDICTION&&) = default;

//------------------------------------------------------------------------------

THE_TYPE
void PREDICTION::init(cusz_predictortype predictor, size_t x, size_t y, size_t z, bool dbg_print)
{
    pimpl->init(predictor, x, y, z, dbg_print);
}

THE_TYPE
void PREDICTION::init(cusz_predictortype predictor, dim3 xyz, bool dbg_print)
{
    pimpl->init(predictor, xyz, dbg_print);
}

THE_TYPE
void PREDICTION::construct(
    cusz_predictortype predictor,
    dim3 const         len3,
    T*                 in_data,
    T**                out_anchor,
    E**                out_errctrl,
    T**                out_outlier,
    double const       eb,
    int const          radius,
    cudaStream_t       stream)
{
    pimpl->construct(predictor, len3, in_data, out_anchor, out_errctrl, out_outlier, eb, radius, stream);
}

THE_TYPE
void PREDICTION::reconstruct(
    cusz_predictortype predictor,
    dim3               len3,
    T*                 in_outlier__out_xdata,
    T*                 in_anchor,
    E*                 in_errctrl,
    // T*                 out_xdata,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    pimpl->reconstruct(predictor, len3, in_outlier__out_xdata, in_anchor, in_errctrl, eb, radius, stream);
}

THE_TYPE
void PREDICTION::clear_buffer() { pimpl->clear_buffer(); }

// getter
THE_TYPE
float PREDICTION::get_time_elapsed() const { return pimpl->get_time_elapsed(); }

THE_TYPE
size_t PREDICTION::get_alloclen_data() const { return pimpl->get_alloclen_data(); }

THE_TYPE
size_t PREDICTION::get_alloclen_quant() const { return pimpl->get_alloclen_quant(); }

THE_TYPE
size_t PREDICTION::get_len_data() const { return pimpl->get_len_data(); }

THE_TYPE
size_t PREDICTION::get_len_quant() const { return pimpl->get_len_quant(); }

THE_TYPE
size_t PREDICTION::get_len_anchor() const { return pimpl->get_len_anchor(); }

THE_TYPE
E* PREDICTION::expose_quant() const { return pimpl->expose_quant(); }

THE_TYPE
E* PREDICTION::expose_errctrl() const { return pimpl->expose_errctrl(); }

THE_TYPE
T* PREDICTION::expose_anchor() const { return pimpl->expose_anchor(); }

THE_TYPE
T* PREDICTION::expose_outlier() const { return pimpl->expose_outlier(); }

}  // namespace cusz

template struct cusz::PredictionUnified<float, uint16_t, float>;
template struct cusz::PredictionUnified<float, uint32_t, float>;
template struct cusz::PredictionUnified<float, float, float>;

// template struct cusz::PredictorSpline3<float, uint32_t, float>;
// template struct cusz::PredictorSpline3<float, float, float>;

#undef THE_TYPE
#undef IMPL