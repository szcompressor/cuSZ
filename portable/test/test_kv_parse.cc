#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "detail/kv_parse.hh"

using _portable::detail::is_kv_pair;
using _portable::detail::parse_strlist;
using _portable::detail::parse_xyz;
using _portable::detail::parse_zyx;
using _portable::detail::separate_kv;

int main()
{
  // ── parse_xyz: 3-D ──────────────────────────────────────────────────
  {
    auto r = parse_xyz("3x4x5");
    assert(r.ndim == 3);
    assert(r.len.x == 3 and r.len.y == 4 and r.len.z == 5);
  }
  // Other delimiters: comma, dash, asterisk, 'm'
  {
    auto r = parse_xyz("3,4,5");
    assert(r.ndim == 3 and r.len.x == 3 and r.len.y == 4 and r.len.z == 5);
  }
  {
    auto r = parse_xyz("3-4-5");
    assert(r.ndim == 3 and r.len.x == 3 and r.len.y == 4 and r.len.z == 5);
  }
  {
    auto r = parse_xyz("3*4*5");
    assert(r.ndim == 3 and r.len.x == 3 and r.len.y == 4 and r.len.z == 5);
  }

  // ── parse_xyz: 2-D (z=1 implied) ────────────────────────────────────
  {
    auto r = parse_xyz("3600x1800");
    assert(r.ndim == 2);
    assert(r.len.x == 3600 and r.len.y == 1800 and r.len.z == 1);
  }

  // ── parse_xyz: 1-D ──────────────────────────────────────────────────
  {
    auto r = parse_xyz("100");
    assert(r.ndim == 1);
    assert(r.len.x == 100 and r.len.y == 1 and r.len.z == 1);
  }

  // ── parse_zyx: reversed order ───────────────────────────────────────
  {
    auto r = parse_zyx("5x4x3");
    assert(r.ndim == 3);
    assert(r.len.x == 3 and r.len.y == 4 and r.len.z == 5);
  }

  // ── is_kv_pair / separate_kv ────────────────────────────────────────
  assert(is_kv_pair("alpha=1.5"));
  assert(not is_kv_pair("bare"));
  assert(not is_kv_pair(""));
  {
    auto kv = separate_kv("alpha=1.5");
    assert(kv.first == "alpha" and kv.second == "1.5");
  }
  {
    // value containing '=' (e.g. base64) — only first '=' splits
    auto kv = separate_kv("blob=a=b");
    assert(kv.first == "blob" and kv.second == "a=b");
  }

  // ── parse_strlist: comma-separated, trimmed ─────────────────────────
  {
    std::vector<std::string> out;
    parse_strlist("a,b,c", out);
    assert(out.size() == 3);
    assert(out[0] == "a" and out[1] == "b" and out[2] == "c");
  }
  {
    std::vector<std::string> out;
    parse_strlist("alpha=1,beta=on,bare", out);
    assert(out.size() == 3);
    assert(out[0] == "alpha=1");
    assert(out[1] == "beta=on");
    assert(out[2] == "bare");
  }
  {
    std::vector<std::string> out;
    parse_strlist("", out);
    assert(out.empty());
  }

  printf("test_kv_parse: PASS\n");
  return 0;
}
