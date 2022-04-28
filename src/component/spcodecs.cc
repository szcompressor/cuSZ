/**
 * @file spcodecs.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "component/spcodecs.hh"
#include <cuda_runtime.h>

namespace cusz {

/*******************************************************************************
 * sparsity-aware coder/decoder, matrix
 *******************************************************************************/

template <typename T, typename M>
SpcodecCSR<T, M>::~SpcodecCSR()
{
    pimpl.reset();
}

template <typename T, typename M>
SpcodecCSR<T, M>::SpcodecCSR() : pimpl{std::make_unique<impl>()}
{
}

template <typename T, typename M>
SpcodecCSR<T, M>::SpcodecCSR(const SpcodecCSR<T, M>& old) : pimpl(new impl(*old.pimpl))
{
    // TODO allocation/deep copy
}

template <typename T, typename M>
SpcodecCSR<T, M>& SpcodecCSR<T, M>::operator=(const SpcodecCSR<T, M>& old)
{
    *pimpl = *old.pimpl;
    // TODO allocation/deep copy
    return *this;
}

template <typename T, typename M>
SpcodecCSR<T, M>::SpcodecCSR(SpcodecCSR<T, M>&& codec) = default;

template <typename T, typename M>
SpcodecCSR<T, M>& SpcodecCSR<T, M>::operator=(SpcodecCSR<T, M>&&) = default;

//------------------------------------------------------------------------------

template <typename T, typename M>
void SpcodecCSR<T, M>::init(size_t const len, int density_factor, bool dbg_print)
{
    pimpl->init(len, density_factor, dbg_print);
}

template <typename T, typename M>
void SpcodecCSR<T, M>::encode(
    T*           in,
    size_t const in_len,
    BYTE*&       out,
    size_t&      out_len,
    cudaStream_t stream,
    bool         dbg_print)
{
    pimpl->encode(in, in_len, out, out_len, stream, dbg_print);
}

template <typename T, typename M>
void SpcodecCSR<T, M>::decode(BYTE* coded, T* decoded, cudaStream_t stream)
{
    pimpl->decode(coded, decoded, stream);
}

template <typename T, typename M>
void SpcodecCSR<T, M>::clear_buffer()
{
    pimpl->clear_buffer();
}

template <typename T, typename M>
float SpcodecCSR<T, M>::get_time_elapsed() const
{
    return pimpl->get_time_elapsed();
}

/*******************************************************************************
 * sparsity-aware coder/decoder, vector
 *******************************************************************************/

template <typename T, typename M>
SpcodecVec<T, M>::~SpcodecVec()
{
    pimpl.reset();
}

template <typename T, typename M>
SpcodecVec<T, M>::SpcodecVec() : pimpl{std::make_unique<impl>()}
{
}

template <typename T, typename M>
SpcodecVec<T, M>::SpcodecVec(const SpcodecVec<T, M>& old) : pimpl(new impl(*old.pimpl))
{
    // TODO allocation/deep copy
}

template <typename T, typename M>
SpcodecVec<T, M>& SpcodecVec<T, M>::operator=(const SpcodecVec<T, M>& old)
{
    *pimpl = *old.pimpl;
    // TODO allocation/deep copy
    return *this;
}

template <typename T, typename M>
SpcodecVec<T, M>::SpcodecVec(SpcodecVec<T, M>&& codec) = default;

template <typename T, typename M>
SpcodecVec<T, M>& SpcodecVec<T, M>::operator=(SpcodecVec<T, M>&&) = default;

//------------------------------------------------------------------------------

template <typename T, typename M>
void SpcodecVec<T, M>::init(size_t const len, int density_factor, bool dbg_print)
{
    pimpl->init(len, density_factor, dbg_print);
}

template <typename T, typename M>
void SpcodecVec<T, M>::encode(
    T*           in,
    size_t const in_len,
    BYTE*&       out,
    size_t&      out_len,
    cudaStream_t stream,
    bool         dbg_print)
{
    pimpl->encode(in, in_len, out, out_len, stream, dbg_print);
}

template <typename T, typename M>
void SpcodecVec<T, M>::decode(BYTE* coded, T* decoded, cudaStream_t stream)
{
    pimpl->decode(coded, decoded, stream);
}

template <typename T, typename M>
void SpcodecVec<T, M>::clear_buffer()
{
    pimpl->clear_buffer();
}

template <typename T, typename M>
float SpcodecVec<T, M>::get_time_elapsed() const
{
    return pimpl->get_time_elapsed();
}

}  // namespace cusz

template class cusz::SpcodecCSR<float, uint32_t>;
template class cusz::SpcodecVec<float, uint32_t>;
