/**
 * @file log.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D531A6F0_2B39_4DF3_8255_26186BD0E575
#define D531A6F0_2B39_4DF3_8255_26186BD0E575

/**
 * @brief Internal (low-level) macros starting with `__PSZ` only define basic
 * strings and how they are printed. The verbosity control (by `PSZ_DBG_ON`)
 * otherwise applies to high-level `PSZ*` macro-defined functions.
 *
 * On the other hand, verbosity control does not apply to debugging/sanitizing
 * functions but to the macro-defined wrappers, e.g., functions in
 * log/sanitize.hh
 *
 */

#include "log/dbg.hh"
// #include "log/log.hh"
#include "log/sanitize.hh"

#endif /* D531A6F0_2B39_4DF3_8255_26186BD0E575 */
