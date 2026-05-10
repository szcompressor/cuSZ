#ifndef _PORTABLE_DETAIL_OPTION_VALUE_HH
#define _PORTABLE_DETAIL_OPTION_VALUE_HH

#ifdef __CUDACC__
#error "detail/option_value.hh must not be included from CUDA translation units"
#endif

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "c_type.h"

namespace _portable::detail {

enum class opt_kind { flag, integer, number, string, dim3, positional };

using opt_value = std::variant<bool, i8, f8, std::string, _portable_len3>;

struct arg_def {
  std::string              name;
  opt_kind                 kind;
  std::vector<std::string> aliases;
  std::optional<opt_value> default_val;  // nullopt = required (positional / dim3)
  std::string              doc;
};

struct arg_store {
  std::unordered_map<std::string, opt_value> values;
  std::unordered_map<std::string, bool>      explicitly_set;
  std::vector<std::string>                   positionals;
};

}  // namespace _portable::detail

#endif  // _PORTABLE_DETAIL_OPTION_VALUE_HH
