#include "memobj_impl.inl"
using namespace portable;
#define INS(T)                                     \
  template class pszmem_cxx<T>;                    \
  template dim3 pszmem_cxx<T>::len3<dim3>() const; \
  template dim3 pszmem_cxx<T>::st3<dim3>() const;

INS(i1)
INS(i2)
INS(i4)
INS(i8)

#undef INS
