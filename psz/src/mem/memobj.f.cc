#include "memobj_impl.inl"

#define INS(T)                                     \
  template class pszmem_cxx<T>;                    \
  template dim3 pszmem_cxx<T>::len3<dim3>() const; \
  template dim3 pszmem_cxx<T>::st3<dim3>() const;

INS(f4)
INS(f8)

#undef INS