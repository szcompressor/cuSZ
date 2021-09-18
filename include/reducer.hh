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

class ReducerAbstraction {
   public:
    virtual void encode() = 0;
    virtual void decode() = 0;
};

template <typename T>
class OneCallAbstraction {
   public:
    virtual void __compress_time()   = 0;
    virtual void __decompress_time() = 0;
};

class OneCallGatherScatter {
   public:
    virtual float get_time_elapsed() const = 0;
    virtual void  gather()                 = 0;
    virtual void  scatter()                = 0;
};

template <typename T>
class SecondPassOnlyAbstraction {
   public:
    virtual void __compress_time_pass_with_metadata() = 0;
    virtual void __decompress_time()                  = 0;
};

template <typename T1, typename T2>
class TwoPassAbstraction {
   public:
    virtual float get_time_elapsed() const = 0;

    virtual void __compress_time_pass1(T1* in, uint32_t& precise_nbyte) = 0;
    virtual void __compress_time_pass2(T1* in, T2* out)                 = 0;
    virtual void __decompress_time(T2* in, T1* out)                     = 0;
};

template <typename T1, typename T2>
class TwoPassCodec : TwoPassAbstraction<T1, T2> {
   public:
    // just a renaming
    virtual void encode_pass1(T1* in, uint32_t& precise_nbyte) = 0;
    virtual void encode_pass2(T1* in, T2* out)                 = 0;
    virtual void decode(T2* in, T1* out)                       = 0;
};

/*
template <typename T>
class TwoPassGatherScatter : TwoPassAbstraction<T> {
   public:
    // just a renaming
    virtual void gather_pass1(T* in, uint32_t& precise_nbyte) = 0;
    virtual void gather_pass2(T* in)                          = 0;
    virtual void scatter()                                    = 0;
};
*/

}  // namespace cusz

#endif