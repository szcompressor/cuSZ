#ifndef AE6DCA2E_F19B_41DB_80CB_11230E548F92
#define AE6DCA2E_F19B_41DB_80CB_11230E548F92

#if defined(PSZ_USE_CUDA)
#include "err_cu.hh"
#elif defined(PSZ_USE_HIP)
#include "err_hip.hh"
#elif defined(PSZ_USE_1API)
// #include "err_1api.hh"
#endif

#endif /* AE6DCA2E_F19B_41DB_80CB_11230E548F92 */
