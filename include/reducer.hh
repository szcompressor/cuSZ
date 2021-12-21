/**
 * @file reducer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef REDUCER_HH
#define REDUCER_HH

namespace cusz {

class VirtualGatherScatter {
   public:
    virtual float get_time_elapsed() const = 0;
    virtual void  gather()                 = 0;
    virtual void  scatter()                = 0;
};

class VariableRate {
   public:
    virtual float get_time_elapsed() const = 0;

   public:
    virtual size_t get_workspace_nbyte(size_t based_inlen) const  = 0;
    virtual size_t get_max_output_nbyte(size_t based_inlen) const = 0;
};

}  // namespace cusz

#endif