#include "utils/synth.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace _ptb::testutils {

uint32_t xorshift32(uint32_t& state)
{
  uint32_t x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return state = x;
}

Synth Synth::parse(const std::string& spec)
{
  Synth       s;
  std::size_t colon = spec.find(':');
  s.mode            = (colon == std::string::npos) ? spec : spec.substr(0, colon);
  if (colon == std::string::npos) return s;

  std::string rest = spec.substr(colon + 1);
  while (not rest.empty()) {
    std::size_t next = rest.find(':');
    std::string kv   = (next == std::string::npos) ? rest : rest.substr(0, next);
    rest             = (next == std::string::npos) ? "" : rest.substr(next + 1);
    std::size_t eq   = kv.find('=');
    if (eq == std::string::npos) continue;
    std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    if (k == "peak")
      s.peak = std::stod(v);
    else if (k == "gamma")
      s.gamma = std::stod(v);
    else if (k == "max")
      s.max = (uint32_t)std::stoul(v);
    else if (k == "seed")
      s.seed = (uint32_t)std::stoul(v);
  }
  return s;
}

namespace {

// cauchy: Cauchy(peak, gamma) rounded + clamped to [0, 2*peak); uniform: flat [0, max).
template <typename E>
void generate(const Synth& s, E* buf, std::size_t len)
{
  uint32_t state = s.seed ? s.seed : 43u;
  if (s.mode == "cauchy") {
    constexpr double pi = 3.14159265358979323846;
    const double     lo = 0.0, hi = 2.0 * s.peak - 1.0;
    for (std::size_t i = 0; i < len; i++) {
      double u = (xorshift32(state) + 0.5) / 4294967296.0;  // (0,1)
      double x = std::round(s.peak + s.gamma * std::tan(pi * (u - 0.5)));
      buf[i]   = (E)(x < lo ? lo : x > hi ? hi : x);
    }
  }
  else if (s.mode == "uniform" or s.mode == "uniform-wide") {
    auto m = s.max ? s.max : 256u;
    for (std::size_t i = 0; i < len; i++) buf[i] = (E)(xorshift32(state) % m);
  }
  else {
    fprintf(stderr, "[synth] unknown mode: %s\n", s.mode.c_str());
    std::exit(2);
  }
}

}  // namespace

void Synth::fill(void* buf, std::size_t len, _ptb_dtype dt) const
{
  switch (dt) {
    case U1: generate(*this, (u1*)buf, len); break;
    case U2: generate(*this, (u2*)buf, len); break;
    case U4: generate(*this, (u4*)buf, len); break;
    default:
      fprintf(stderr, "[synth] unsupported dtype %d (use u1|u2|u4)\n", (int)dt);
      std::exit(2);
  }
}

double Synth::pmf1() const { return pmf1_from(gamma); }

double Synth::pmf1_from(double gamma)
{
  constexpr double pi = 3.14159265358979323846;
  return (2.0 / pi) * std::atan(0.5 / gamma);
}

double Synth::gamma_from(double pmf1)
{
  constexpr double pi = 3.14159265358979323846;
  return 0.5 / std::tan(pi * pmf1 / 2.0);
}

}  // namespace _ptb::testutils
