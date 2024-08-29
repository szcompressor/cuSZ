#include "l23_x.cuhip.inl"

INSTANTIATE_GPU_L23X_1param(f4);
INSTANTIATE_GPU_L23X_1param(f8);

#undef INSTANTIATE_GPU_L23X_1param
#undef INSTANTIATE_GPU_L23X_2params
#undef INSTANTIATE_GPU_L23X_3params