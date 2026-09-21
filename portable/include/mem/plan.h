#ifndef _PORTABLE_MEM_PLAN_H
#define _PORTABLE_MEM_PLAN_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "../c_type.h"
#include "../cxx_typing.h"

namespace _ptb {

class mem_plan {
 public:
  struct _frame {
    i2 const     sym;
    i2 const     alias_of;
    u1 const     dtype;  // _ptb_dtype
    size_t const len;

    _frame(int _sym, _ptb_dtype _t, size_t _len, int _alias_of = NO_ALIAS) :
        sym(_sym), alias_of(_alias_of), dtype(_t), len(_len)
    {
    }
  };

  struct tables {
    std::vector<_frame> data, state;
  };

  static constexpr size_t _0x0100B = 0x0100;  // $\sim $cudaMalloc
  static constexpr size_t npos     = std::numeric_limits<size_t>::max();
  static constexpr int    NO_ALIAS = -1;

  class region {
   public:
    size_t n_frame() const { return _v_frame.size(); }
    size_t bytes() const { return _total_aligned_bytes; }

    int sym(size_t i) const { return _v_frame[i].sym; }
    int alias_of(size_t i) const { return _v_frame[i].alias_of; }
    bool owns(size_t i) const { return _v_frame[i].alias_of == NO_ALIAS; }

    std::vector<size_t> const& v_nbyte() const { return _v_nbyte; }
    std::vector<size_t> const& v_entry() const { return _v_entry; }
    std::vector<void*> const& v_ptr() const { return _v_ptr; }

    size_t nbyte(size_t i) const { return _v_nbyte[i]; }
    size_t entry(size_t i) const { return _v_entry[i]; }

    template <typename T>
    T* ptr(size_t i) const
    { return (T*)_v_ptr[i]; }

    size_t find(int sym) const
    {
      return (sym >= 0 and (size_t) sym < _sym2idx.size() and _sym2idx[sym] >= 0)
                 ? (size_t)_sym2idx[sym]
                 : npos;
    }

    template <typename T>
    T* ptr_of(int sym) const
    {
      auto const i = find(sym);
      return i == npos ? nullptr : (T*)_v_ptr[i];
    }

   private:
    friend class mem_plan;

    _ptb_errno consolidate(std::vector<_frame> frames)
    {
      _v_frame = std::move(frames);

      auto   status = _SUCCESS;
      size_t cursor = 0;

      int max_sym = -1;
      for (auto const& f : _v_frame)
        if (f.sym > max_sym) max_sym = f.sym;
      _sym2idx.assign((size_t)(max_sym + 1), -1);

      for (size_t i = 0; i < _v_frame.size(); i++) {
        if (_v_frame[i].sym < 0 or find(_v_frame[i].sym) != npos)
          status = _FAIL_GENERAL;
        else
          _sym2idx[_v_frame[i].sym] = (i2)i;

        auto const need = _v_frame[i].len * dtype_width((_ptb_dtype)_v_frame[i].dtype);

        if (_v_frame[i].alias_of == NO_ALIAS) {
          _v_nbyte.push_back(need);
          _v_entry.push_back(cursor);
          cursor += align_0x100(need);
          continue;
        }

        auto const t = find(_v_frame[i].alias_of);
        if (t == npos or t >= i or not owns(t) or need > _v_nbyte[t]) {
          status = _FAIL_GENERAL;
          _v_nbyte.push_back(need);
          _v_entry.push_back(npos);
        }
        else {
          _v_nbyte.push_back(need ? need : _v_nbyte[t]);
          _v_entry.push_back(_v_entry[t]);
        }
      }

      _v_ptr.assign(_v_frame.size(), nullptr);
      _total_aligned_bytes = cursor;
      return status;
    }

    void set_base(void* base)
    {
      for (size_t i = 0; i < _v_frame.size(); i++)
        _v_ptr[i] = (_v_nbyte[i] == 0 or _v_entry[i] == npos) ? nullptr : (u1*)base + _v_entry[i];
    }

    std::vector<_frame> _v_frame;
    std::vector<i2>     _sym2idx;
    std::vector<size_t> _v_nbyte, _v_entry;
    std::vector<void*>  _v_ptr;
    size_t              _total_aligned_bytes = 0;
  };

  explicit mem_plan(tables t) : mem_plan(std::move(t.data), std::move(t.state)) {}

  explicit mem_plan(std::vector<_frame> data_frames, std::vector<_frame> state_frames = {})
  {
    auto const d = _data.consolidate(std::move(data_frames));
    auto const s = _state.consolidate(std::move(state_frames));
    if (d != _SUCCESS or s != _SUCCESS) _status = _FAIL_GENERAL;
  }

  struct total {
    size_t data, state;
  };

  operator total() const { return {_data.bytes(), _state.bytes()}; }

  friend total operator+(total a, total b) { return {a.data + b.data, a.state + b.state}; }

  friend total operator|(total a, total b)
  { return {a.data > b.data ? a.data : b.data, a.state > b.state ? a.state : b.state}; }

  class tree {
   public:
    using attach_fn = std::function<void(void*, void*)>;

    tree(total leaf, attach_fn attach = nullptr) : _bytes(leaf), _attach(std::move(attach)) {}

    total bytes() const { return _bytes; }

    friend tree operator+(tree a, tree b)
    { return combine(kind::AND, std::move(a), std::move(b)); }
    friend tree operator|(tree a, tree b) { return combine(kind::OR, std::move(a), std::move(b)); }

    std::vector<total> assign(total base = {0, 0}, void* d_data = nullptr, void* d_state = nullptr) const
    {
      std::vector<total> out;
      assign_into(base, (u1*)d_data, (u1*)d_state, out);
      return out;
    }

   private:
    enum class kind { LEAF, AND, OR };

    tree() = default;

    static tree combine(kind k, tree a, tree b)
    {
      tree out;
      out._kind  = k;
      out._bytes = k == kind::AND ? (a._bytes + b._bytes) : (a._bytes | b._bytes);
      out._kids  = a._kind == k ? std::move(a._kids) : std::vector<tree>{std::move(a)};
      if (b._kind == k)
        for (auto& kid : b._kids) out._kids.push_back(std::move(kid));
      else
        out._kids.push_back(std::move(b));
      return out;
    }

    void assign_into(total base, u1* d_data, u1* d_state, std::vector<total>& out) const
    {
      if (_kind == kind::LEAF) {
        out.push_back(base);
        if (_attach) _attach(d_data + base.data, d_state + base.state);
        return;
      }
      if (_kind == kind::OR) {
        for (auto const& kid : _kids) kid.assign_into(base, d_data, d_state, out);
        return;
      }
      total cursor = base;
      for (auto const& kid : _kids) {
        kid.assign_into(cursor, d_data, d_state, out);
        cursor = cursor + kid.bytes();
      }
    }

    kind              _kind = kind::LEAF;
    total             _bytes{0, 0};
    attach_fn         _attach;
    std::vector<tree> _kids;
  };

  void set_base(void* d_data, void* d_state)
  {
    _data.set_base(d_data);
    _state.set_base(d_state);
  }

  region const& data() const { return _data; }
  region const& state() const { return _state; }
  _ptb_errno status() const { return _status; }

 private:
  static constexpr size_t align_0x100(size_t n)
  { return (n + _0x0100B - 1) / _0x0100B * _0x0100B; }

  region     _data, _state;
  _ptb_errno _status = _SUCCESS;
};

class buf_base : public mem_plan {
 public:
  using mem_plan::mem_plan;
  virtual ~buf_base() = default;

  void attach(void* d_data, void* d_state)
  {
    set_base(d_data, d_state);
    init_state();
  }

  virtual void init()              = 0;  // can use standalone
  virtual void reset(void* stream) = 0;

 protected:
  virtual void init_state() {}
};

inline mem_plan::tree leaf(buf_base& b)
{ return mem_plan::tree((mem_plan::total)b, [&b](void* d, void* s) { b.attach(d, s); }); }

}  // namespace _ptb

#endif /* _PORTABLE_MEM_PLAN_H */
