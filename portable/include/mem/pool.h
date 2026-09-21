#ifndef _PORTABLE_MEM_POOL_H
#define _PORTABLE_MEM_POOL_H

#include "cxx_backends.h"
#include "plan.h"

namespace _ptb {

class pool {
 public:
  void allocate(mem_plan::total const& bytes)
  {
    if (bytes.data) _d_data = MAKE_UNIQUE_DEVICE(u1, bytes.data);
    if (bytes.state) _d_state = MAKE_UNIQUE_DEVICE(u1, bytes.state);
  }

  void* data() const { return _d_data.get(); }
  void* state() const { return _d_state.get(); }

 private:
  GPU_unique_dptr<u1[]> _d_data, _d_state;
};

}  // namespace _ptb

#endif /* _PORTABLE_MEM_POOL_H */
