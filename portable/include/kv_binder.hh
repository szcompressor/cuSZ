#ifndef _PORTABLE_KV_BINDER_HH
#define _PORTABLE_KV_BINDER_HH

#ifdef __CUDACC__
#error "kv_binder.hh must not be included from CUDA translation units"
#endif

#include <functional>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

#include "detail/kv_parse.hh"
#include "detail/str2num.hh"

namespace _portable {

// Binds a comma-separated key=value (or bare-key) string into a typed struct T.
// Sits on top of detail::parse_strlist + detail::separate_kv.
//
// Usage:
//   static const auto binder = _portable::kv_binder<MyStruct>()
//     .number({"alpha", "intp-alpha"}, &MyStruct::alpha)
//     .flag({"enabled"},               &MyStruct::enabled)
//     .flag_ref({"arr_0","a0"}, [](MyStruct& s) -> bool& { return s.arr[0]; })
//     .custom({"mode"}, [](MyStruct& s, const std::string& v) { ... });
//   binder.bind("alpha=1.5,enabled", target);

template <typename T>
class kv_binder {
  using handler_fn  = std::function<void(T&, const std::string&)>;
  using bool_ref_fn = std::function<bool&(T&)>;

  struct entry {
    std::vector<std::string> keys;
    handler_fn               handler;
  };

  std::vector<entry> entries_;

  kv_binder&& add(std::initializer_list<const char*> keys, handler_fn h)
  {
    entry e;
    for (auto* k : keys) e.keys.push_back(k);
    e.handler = std::move(h);
    entries_.push_back(std::move(e));
    return std::move(*this);
  }

public:
  // Floating-point field via member pointer.
  template <typename F>
  kv_binder&& number(std::initializer_list<const char*> keys, F T::* field)
  {
    static_assert(std::is_floating_point_v<F>, "number() requires a floating-point field");
    return add(keys, [field](T& t, const std::string& v) {
      auto n = detail::str_to_num(v.c_str());
      if (n) t.*field = static_cast<F>(*n);
    });
  }

  // Bool field via member pointer.
  // Bare key or value "on"/"ON" -> true; "off"/"OFF" -> false.
  kv_binder&& flag(std::initializer_list<const char*> keys, bool T::* field)
  {
    return add(keys, [field](T& t, const std::string& v) {
      t.*field = not (v == "off" or v == "OFF");
    });
  }

  // Bool array element via accessor — member pointer cannot address array elements.
  kv_binder&& flag_ref(std::initializer_list<const char*> keys, bool_ref_fn fn)
  {
    return add(keys, [fn](T& t, const std::string& v) {
      fn(t) = not (v == "off" or v == "OFF");
    });
  }

  // Custom handler — full control over parsing and assignment.
  kv_binder&& custom(std::initializer_list<const char*> keys, handler_fn h)
  {
    return add(keys, std::move(h));
  }

  // Bind comma-separated string into target.
  // Each token is either "key" (bare, value = "") or "key=value".
  void bind(const char* in_str, T& target) const
  {
    std::vector<std::string> tokens;
    detail::parse_strlist(in_str, tokens);

    for (auto& tok : tokens) {
      std::string key, val;
      if (detail::is_kv_pair(tok)) {
        auto kv = detail::separate_kv(tok);
        key = kv.first;
        val = kv.second;
      }
      else {
        key = tok;  // bare key — val stays ""
      }

      for (auto& e : entries_) {
        bool matched = false;
        for (auto& k : e.keys)
          if (k == key) { matched = true; break; }
        if (matched) { e.handler(target, val); break; }
      }
    }
  }
};

}  // namespace _portable

#endif  // _PORTABLE_KV_BINDER_HH
