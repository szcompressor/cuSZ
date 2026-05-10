#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "kv_binder.hh"

using _portable::kv_binder;

struct config {
  double alpha;
  double beta;
  bool   enabled;
  bool   trace;
  bool   arr[4];
  int    mode;
};

int main()
{
  static const auto binder =
      kv_binder<config>()
          .number({"alpha", "intp-alpha"}, &config::alpha)
          .number({"beta"}, &config::beta)
          .flag({"enabled"}, &config::enabled)
          .flag({"trace", "tr"}, &config::trace)
          .flag_ref({"arr0"}, [](config& c) -> bool& { return c.arr[0]; })
          .flag_ref({"arr1"}, [](config& c) -> bool& { return c.arr[1]; })
          .custom({"mode"}, [](config& c, const std::string& v) {
            if (v == "fast")
              c.mode = 1;
            else if (v == "slow")
              c.mode = 2;
            else
              throw std::runtime_error("bad mode value: " + v);
          });

  // ── number + flag (bare key = true) ─────────────────────────────────
  {
    config c{};
    binder.bind("alpha=1.5,beta=2.0,enabled", c);
    assert(c.alpha == 1.5);
    assert(c.beta == 2.0);
    assert(c.enabled == true);
  }

  // ── flag with explicit on/off ───────────────────────────────────────
  {
    config c{.enabled = true, .trace = true};
    binder.bind("enabled=off,trace=ON", c);
    assert(c.enabled == false);
    assert(c.trace == true);  // ON should still mean true
  }

  // ── alias resolution ────────────────────────────────────────────────
  {
    config c{};
    binder.bind("intp-alpha=3.14", c);
    assert(c.alpha == 3.14);
  }

  // ── flag_ref into array element ─────────────────────────────────────
  {
    config c{};
    binder.bind("arr0,arr1=off", c);
    assert(c.arr[0] == true);
    assert(c.arr[1] == false);
    assert(c.arr[2] == false);  // untouched
    assert(c.arr[3] == false);
  }

  // ── custom handler ──────────────────────────────────────────────────
  {
    config c{};
    binder.bind("mode=fast", c);
    assert(c.mode == 1);
    binder.bind("mode=slow", c);
    assert(c.mode == 2);
  }

  // ── custom handler throws on bad value ──────────────────────────────
  {
    config c{};
    bool   threw = false;
    try {
      binder.bind("mode=banana", c);
    }
    catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  // ── Unknown keys are silently ignored (per spec) ────────────────────
  {
    config c{};
    binder.bind("unknown_key=42,alpha=1.0", c);
    assert(c.alpha == 1.0);  // alpha still set
  }

  // ── Empty input is a no-op ──────────────────────────────────────────
  {
    config c{.alpha = 99.0};
    binder.bind("", c);
    assert(c.alpha == 99.0);
  }

  printf("test_kv_binder: PASS\n");
  return 0;
}
