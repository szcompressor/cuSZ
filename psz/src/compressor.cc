
#include "compressor.inl"

template class psz::Compressor<f4>;
template class psz::Compressor<f8>;

template struct psz::compression_pipeline<f4, u2>;
template struct psz::compression_pipeline<f8, u2>;
