// SPDX-License-Identifier: BSD-3-Clause
// Header-only TOML-subset parser for the psz test-data registry.
// See doc/2026-05-09_test-infra-migration-bin_pred.md for the registry shape
// and rationale.
//
// Subset supported:
//   - blank lines and `# comment` lines (full-line only)
//   - `[stanza.name]` headers (dots in name allowed; no nesting)
//   - `key = value` lines inside a stanza
//   - value forms:
//       quoted string:  "..."       (no escape processing — raw chars between quotes)
//       integer:        42 / -7
//       float:          3.14 / 1e-4 / -1.0e7
//       bool:           true / false
//       array:          [a, b, c]   (any of the above; nested arrays not supported)
//
// Not supported (deliberate): nested stanzas, inline tables, multi-line
// strings, string escapes, datetime, hex/octal/bin integers. If we ever need
// any of those, swap to a real TOML library at that point.

#ifndef PSZ_TEST_LIB_TOML_LITE_HH
#define PSZ_TEST_LIB_TOML_LITE_HH

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace psz_test {

// A Stanza maps key -> raw value string (after stripping quotes / brackets).
// Callers parse to int/double/array via the helpers below.
struct Stanza {
  std::unordered_map<std::string, std::string> raw;

  bool has(const std::string& key) const { return raw.count(key) != 0; }
  const std::string& at(const std::string& key) const { return raw.at(key); }
};

// Registry maps stanza-name (e.g. "rtm.0480") -> Stanza.
struct Registry {
  std::unordered_map<std::string, Stanza> stanzas;

  bool has(const std::string& name) const { return stanzas.count(name) != 0; }
  const Stanza* get(const std::string& name) const
  {
    auto it = stanzas.find(name);
    return (it == stanzas.end()) ? nullptr : &it->second;
  }
};

// ---- helpers ---------------------------------------------------------------

inline std::string strip(std::string s)
{
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(a, b - a);
}

inline std::string strip_quotes(std::string v)
{
  v = strip(v);
  if (v.size() >= 2 && v.front() == '"' && v.back() == '"') return v.substr(1, v.size() - 2);
  return v;
}

inline int64_t as_int(const Stanza& s, const std::string& key)
{
  return std::stoll(s.at(key));
}
inline double as_double(const Stanza& s, const std::string& key)
{
  return std::stod(s.at(key));
}
inline std::string as_string(const Stanza& s, const std::string& key)
{
  return strip_quotes(s.at(key));
}

// Parse "[a, b, c]" -> {"a", "b", "c"} (whitespace and quotes stripped).
inline std::vector<std::string> as_string_array(const Stanza& s, const std::string& key)
{
  std::vector<std::string> out;
  std::string v = strip(s.at(key));
  if (v.size() < 2 || v.front() != '[' || v.back() != ']')
    throw std::runtime_error("toml_lite: expected array for key '" + key + "': " + v);
  std::string inner = v.substr(1, v.size() - 2);
  std::string item;
  std::stringstream ss(inner);
  while (std::getline(ss, item, ',')) out.push_back(strip_quotes(item));
  return out;
}

inline std::vector<int64_t> as_int_array(const Stanza& s, const std::string& key)
{
  std::vector<int64_t> out;
  for (auto& v : as_string_array(s, key)) out.push_back(std::stoll(v));
  return out;
}

// ---- parser ----------------------------------------------------------------

inline Registry parse_string(const std::string& text)
{
  Registry reg;
  Stanza* cur = nullptr;
  std::string cur_name;
  size_t lineno = 0;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    ++lineno;
    std::string s = strip(line);
    if (s.empty() || s[0] == '#') continue;

    if (s.front() == '[' && s.back() == ']') {
      cur_name = s.substr(1, s.size() - 2);
      cur = &reg.stanzas[cur_name];
      continue;
    }

    auto eq = s.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error(
          "toml_lite: line " + std::to_string(lineno) + ": expected '=': " + line);
    if (cur == nullptr)
      throw std::runtime_error(
          "toml_lite: line " + std::to_string(lineno) +
          ": key/value before any [stanza]: " + line);

    std::string key = strip(s.substr(0, eq));
    std::string val = strip(s.substr(eq + 1));
    // strip trailing inline `# comment` only when not inside a quoted string
    bool in_quote = false;
    for (size_t i = 0; i < val.size(); ++i) {
      if (val[i] == '"') in_quote = !in_quote;
      else if (val[i] == '#' && !in_quote) {
        val = strip(val.substr(0, i));
        break;
      }
    }
    cur->raw[key] = val;
  }
  return reg;
}

inline Registry parse_file(const std::string& path)
{
  std::ifstream f(path);
  if (!f) throw std::runtime_error("toml_lite: cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  return parse_string(buf.str());
}

// ---- registry-specific helpers --------------------------------------------

// Resolve the registry path with this priority:
//   1. explicit `path_override` if non-empty
//   2. `$PSZ_TEST_DATA` env var if set
//   3. `$HOME/.psz_test_data.toml`
// Returns empty string if none of the above resolves to a readable file.
inline std::string resolve_registry_path(const std::string& path_override = "")
{
  auto exists = [](const std::string& p) {
    if (p.empty()) return false;
    std::ifstream f(p);
    return (bool)f;
  };
  if (exists(path_override)) return path_override;
  if (const char* env = std::getenv("PSZ_TEST_DATA")) {
    std::string p(env);
    if (exists(p)) return p;
  }
  if (const char* home = std::getenv("HOME")) {
    std::string p = std::string(home) + "/.psz_test_data.toml";
    if (exists(p)) return p;
  }
  return "";
}

// One-stop dataset descriptor as parsed from a stanza.
struct DatasetEntry {
  std::string path;
  int64_t x = 0, y = 0, z = 1;
  double eb = 0.0;
  std::string klass;  // e.g. "3d_seismic"
};

inline std::optional<DatasetEntry> lookup_dataset(
    const Registry& reg, const std::string& name)
{
  const Stanza* st = reg.get(name);
  if (!st) return std::nullopt;
  DatasetEntry d;
  d.path = as_string(*st, "path");
  auto dims = as_int_array(*st, "dims");
  if (dims.size() >= 1) d.x = dims[0];
  if (dims.size() >= 2) d.y = dims[1];
  if (dims.size() >= 3) d.z = dims[2];
  if (st->has("eb")) d.eb = as_double(*st, "eb");
  if (st->has("class")) d.klass = as_string(*st, "class");
  return d;
}

inline bool file_exists(const std::string& path)
{
  std::ifstream f(path);
  return (bool)f;
}

}  // namespace psz_test

#endif
