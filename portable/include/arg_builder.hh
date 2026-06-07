#ifndef _PORTABLE_ARG_BUILDER_HH
#define _PORTABLE_ARG_BUILDER_HH

#ifdef __CUDACC__
#error "arg_builder.hh must not be included from CUDA translation units"
#endif

#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "c_type.h"
#include "detail/kv_parse.hh"
#include "detail/option_value.hh"
#include "detail/str2num.hh"

namespace _ptb {

//------------------------------------------------------------------------------
// arg_result: owns all parsed values; returned by value from arg_builder::parse().
//------------------------------------------------------------------------------

struct arg_result {
  template <typename T>
  T get(const char* name) const
  {
    auto it = store_.values.find(name);
    if (it == store_.values.end())
      throw std::runtime_error(std::string("unknown option: ") + name);
    if (not std::holds_alternative<T>(it->second))
      throw std::runtime_error(std::string("type mismatch for option: ") + name);
    return std::get<T>(it->second);
  }

  bool is_set(const char* name) const
  {
    auto it = store_.explicitly_set.find(name);
    return it != store_.explicitly_set.end() and it->second;
  }

  const std::string& positional(int i) const
  {
    if (i < 0 or i >= static_cast<int>(store_.positionals.size()))
      throw std::runtime_error("positional index out of range");
    return store_.positionals[i];
  }

  int positional_count() const { return static_cast<int>(store_.positionals.size()); }

  void print(FILE* out = stdout) const
  {
    for (auto& [k, v] : store_.values) {
      fprintf(out, "  %-20s = ", k.c_str());
      std::visit(
          [&](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, bool>)
              fprintf(out, "%s\n", val ? "true" : "false");
            else if constexpr (std::is_same_v<T, i8>)
              fprintf(out, "%lld\n", static_cast<long long>(val));
            else if constexpr (std::is_same_v<T, f8>)
              fprintf(out, "%g\n", val);
            else if constexpr (std::is_same_v<T, std::string>)
              fprintf(out, "%s\n", val.c_str());
            else if constexpr (std::is_same_v<T, _ptb_len3>)
              fprintf(out, "%zux%zux%zu\n", val.x, val.y, val.z);
          },
          v);
    }
  }

  detail::arg_store store_;
};

//------------------------------------------------------------------------------
// arg_builder: declare options, then call parse() to obtain an arg_result.
// Designed as a file-scope static — parse() is const and re-entrant.
//
//   static const auto cli = _ptb::arg_builder("bin_phf")
//     .positional("fname",  "input binary file")
//     .dim3("len",          {"-l", "--xyz"},          "data dimensions")
//     .number("eb",         {"-e", "--error-bound"},   1e-3, "error bound")
//     .integer("repeat",    {"-r", "--repeat"},         1,   "repetitions")
//     .flag("verbose",      {"-v"},                         "verbose output")
//     ;
//   int main(int argc, char** argv) {
//     auto args = cli.parse(argc, argv);
//     auto len  = args.get<_ptb_len3>("len");
//     auto eb   = args.get<f8>("eb");
//   }
//------------------------------------------------------------------------------

class arg_builder {
 public:
  explicit arg_builder(const char* prog_name) : prog_name_(prog_name) {}

  arg_builder(arg_builder&&)                 = default;
  arg_builder(const arg_builder&)            = delete;
  arg_builder& operator=(const arg_builder&) = delete;

  arg_builder&& positional(const char* name, const char* doc = "")
  {
    defs_.push_back({name, detail::opt_kind::positional, {}, std::nullopt, doc});
    return std::move(*this);
  }

  arg_builder&& flag(
      const char* name, std::initializer_list<const char*> aliases, const char* doc = "")
  {
    defs_.push_back(
        {name, detail::opt_kind::flag, make_aliases(aliases), detail::opt_value{false}, doc});
    return std::move(*this);
  }

  arg_builder&& integer(
      const char* name, std::initializer_list<const char*> aliases, i8 default_val,
      const char* doc = "")
  {
    defs_.push_back(
        {name, detail::opt_kind::integer, make_aliases(aliases), detail::opt_value{default_val},
         doc});
    return std::move(*this);
  }

  arg_builder&& number(
      const char* name, std::initializer_list<const char*> aliases, f8 default_val,
      const char* doc = "")
  {
    defs_.push_back(
        {name, detail::opt_kind::number, make_aliases(aliases), detail::opt_value{default_val},
         doc});
    return std::move(*this);
  }

  arg_builder&& string(
      const char* name, std::initializer_list<const char*> aliases, const char* default_val,
      const char* doc = "")
  {
    defs_.push_back(
        {name, detail::opt_kind::string, make_aliases(aliases),
         detail::opt_value{std::string(default_val)}, doc});
    return std::move(*this);
  }

  // Dimension triple — required (no default).
  // Accepts: "NxMxK", "N,M,K", "N*M*K", "N-M-K", or plain "N" for 1-D.
  arg_builder&& dim3(
      const char* name, std::initializer_list<const char*> aliases, const char* doc = "")
  {
    defs_.push_back({name, detail::opt_kind::dim3, make_aliases(aliases), std::nullopt, doc});
    return std::move(*this);
  }

  // Parse argv. Calls print_help()+exit(0) on -h/--help.
  // Throws std::runtime_error on bad/missing input.
  arg_result parse(int argc, char** argv) const
  {
    if (argc <= 1) {
      print_help();
      exit(0);
    }

    std::unordered_map<std::string, std::string> alias_map;
    int                                          n_positional_defs = 0;
    for (auto& d : defs_) {
      if (d.kind == detail::opt_kind::positional) {
        ++n_positional_defs;
        continue;
      }
      for (auto& a : d.aliases) alias_map[a] = d.name;
    }

    detail::arg_store store;
    for (auto& d : defs_) {
      if (d.kind == detail::opt_kind::positional) continue;
      if (d.default_val) store.values[d.name] = *d.default_val;
    }

    int positional_idx = 0;
    int i              = 1;
    while (i < argc) {
      std::string tok(argv[i]);

      if (tok == "-h" or tok == "--help") {
        print_help();
        exit(0);
      }

      if (tok[0] == '-') {
        // accept both "--key val" and "--key=val" forms.
        std::string key = tok;
        std::string inline_val;
        bool        has_inline = false;
        auto        eq         = tok.find('=');
        if (eq != std::string::npos) {
          key        = tok.substr(0, eq);
          inline_val = tok.substr(eq + 1);
          has_inline = true;
        }

        auto it = alias_map.find(key);
        if (it == alias_map.end()) throw std::runtime_error("unknown option: " + key);

        const std::string& name = it->second;
        const auto&        def  = find_def(name);

        if (def.kind == detail::opt_kind::flag) {
          if (has_inline) throw std::runtime_error("flag does not take a value: " + key);
          store.values[name]         = true;
          store.explicitly_set[name] = true;
        }
        else {
          const char* val;
          if (has_inline) { val = inline_val.c_str(); }
          else {
            if (i + 1 >= argc) throw std::runtime_error("option " + key + " requires an argument");
            val = argv[++i];
          }

          switch (def.kind) {
            case detail::opt_kind::integer: {
              auto v = detail::str_to_int(val);
              if (not v) throw std::runtime_error("invalid integer for " + tok + ": " + val);
              store.values[name] = *v;
              break;
            }
            case detail::opt_kind::number: {
              auto v = detail::str_to_num(val);
              if (not v) throw std::runtime_error("invalid number for " + tok + ": " + val);
              store.values[name] = *v;
              break;
            }
            case detail::opt_kind::string: store.values[name] = std::string(val); break;
            case detail::opt_kind::dim3: store.values[name] = detail::parse_xyz(val).len; break;
            default: break;
          }
          store.explicitly_set[name] = true;
        }
      }
      else {
        if (positional_idx >= n_positional_defs)
          throw std::runtime_error("unexpected positional argument: " + tok);
        store.positionals.push_back(tok);
        const auto& def                = positional_def_at(positional_idx);
        store.values[def.name]         = tok;
        store.explicitly_set[def.name] = true;
        ++positional_idx;
      }
      ++i;
    }

    for (auto& d : defs_) {
      if (store.values.find(d.name) == store.values.end())
        throw std::runtime_error("required argument not provided: " + d.name);
    }

    arg_result result;
    result.store_ = std::move(store);
    return result;
  }

  void print_help(FILE* out = stdout) const
  {
    fprintf(out, "usage: %s", prog_name_.c_str());
    for (auto& d : defs_)
      if (d.kind == detail::opt_kind::positional) fprintf(out, " <%s>", d.name.c_str());
    fprintf(out, " [options]\n\n");

    bool has_positionals = false;
    for (auto& d : defs_)
      if (d.kind == detail::opt_kind::positional) {
        has_positionals = true;
        break;
      }

    if (has_positionals) {
      fprintf(out, "positional arguments:\n");
      for (auto& d : defs_) {
        if (d.kind != detail::opt_kind::positional) continue;
        fprintf(out, "  %-24s%s\n", d.name.c_str(), d.doc.c_str());
      }
      fprintf(out, "\n");
    }

    fprintf(out, "options:\n");
    fprintf(out, "  -h, --help              show this help\n");
    for (auto& d : defs_) {
      if (d.kind == detail::opt_kind::positional) continue;
      std::string alias_str;
      for (auto& a : d.aliases) {
        if (not alias_str.empty()) alias_str += ", ";
        alias_str += a;
      }
      if (d.kind != detail::opt_kind::flag) alias_str += " <val>";
      fprintf(out, "  %-24s%s", alias_str.c_str(), d.doc.c_str());

      if (d.default_val) {
        fprintf(out, " [default: ");
        std::visit(
            [&](auto&& v) {
              using T = std::decay_t<decltype(v)>;
              if constexpr (std::is_same_v<T, bool>)
                fprintf(out, "%s", v ? "true" : "false");
              else if constexpr (std::is_same_v<T, i8>)
                fprintf(out, "%lld", static_cast<long long>(v));
              else if constexpr (std::is_same_v<T, f8>)
                fprintf(out, "%g", v);
              else if constexpr (std::is_same_v<T, std::string>)
                fprintf(out, "%s", v.c_str());
              else if constexpr (std::is_same_v<T, _ptb_len3>)
                fprintf(out, "%zux%zux%zu", v.x, v.y, v.z);
            },
            *d.default_val);
        fprintf(out, "]");
      }
      fprintf(out, "\n");
    }
  }

 private:
  static std::vector<std::string> make_aliases(std::initializer_list<const char*> aliases)
  {
    std::vector<std::string> v;
    for (auto* a : aliases) v.push_back(a);
    return v;
  }

  const detail::arg_def& find_def(const std::string& name) const
  {
    for (auto& d : defs_)
      if (d.name == name) return d;
    throw std::runtime_error("internal: unknown option name: " + name);
  }

  const detail::arg_def& positional_def_at(int idx) const
  {
    int n = 0;
    for (auto& d : defs_) {
      if (d.kind != detail::opt_kind::positional) continue;
      if (n == idx) return d;
      ++n;
    }
    throw std::runtime_error("internal: positional index out of range");
  }

  std::string                  prog_name_;
  std::vector<detail::arg_def> defs_;
};

}  // namespace _ptb

#endif  // _PORTABLE_ARG_BUILDER_HH
