#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "arg_builder.hh"

using _portable::arg_builder;
using _portable::arg_result;

// Helper: build a fake argv from a vector<string>. The argv pointers are
// non-owning views into the strings; the strings vector must outlive the
// arg_result returned by parse().
struct argv_holder {
  std::vector<std::string> store;
  std::vector<char*>       argv;
  argv_holder(std::initializer_list<const char*> args)
  {
    for (auto* s : args) store.emplace_back(s);
    for (auto& s : store) argv.push_back(const_cast<char*>(s.c_str()));
  }
  int    argc() const { return static_cast<int>(argv.size()); }
  char** data() { return argv.data(); }
};

int main()
{
  // ── Schema reused across cases ──────────────────────────────────────
  static const auto cli =
      arg_builder("test_bin")
          .positional("input", "input file")
          .integer("repeat", {"-r", "--repeat"}, 1, "repeats")
          .number("eb", {"-e", "--eb"}, 1e-3, "error bound")
          .string("mode", {"-m", "--mode"}, "rel", "mode")
          .flag("verbose", {"-v", "--verbose"}, "verbose")
          .dim3("len", {"-l", "--xyz"}, "data dims");

  // ── Parse: defaults applied, only required (positional + dim3) given ─
  {
    argv_holder a{"prog", "in.f4", "-l", "3x4x5"};
    auto        r = cli.parse(a.argc(), a.data());

    assert(r.get<std::string>("input") == "in.f4");
    assert(r.get<i8>("repeat") == 1);                   // default
    auto eb = r.get<f8>("eb");
    assert(eb > 9e-4 and eb < 1.1e-3);                  // default 1e-3
    assert(r.get<std::string>("mode") == "rel");        // default
    assert(r.get<bool>("verbose") == false);            // default

    auto len = r.get<_portable_len3>("len");
    assert(len.x == 3 and len.y == 4 and len.z == 5);

    assert(r.is_set("input"));
    assert(r.is_set("len"));
    assert(not r.is_set("repeat"));
    assert(not r.is_set("verbose"));
  }

  // ── Parse: all options explicit + alias variants ────────────────────
  {
    argv_holder a{
        "prog", "data.f8", "-r", "10", "--eb", "5e-5", "-m", "abs", "-v", "--xyz", "100x200"};
    auto r = cli.parse(a.argc(), a.data());

    assert(r.get<std::string>("input") == "data.f8");
    assert(r.get<i8>("repeat") == 10);
    auto eb = r.get<f8>("eb");
    assert(eb > 4.9e-5 and eb < 5.1e-5);
    assert(r.get<std::string>("mode") == "abs");
    assert(r.get<bool>("verbose") == true);

    auto len = r.get<_portable_len3>("len");
    assert(len.x == 100 and len.y == 200 and len.z == 1);  // 2-D, z=1

    assert(r.is_set("repeat"));
    assert(r.is_set("eb"));
    assert(r.is_set("verbose"));
  }

  // ── Error: unknown option ───────────────────────────────────────────
  {
    argv_holder a{"prog", "in.f4", "-l", "3x4x5", "--bogus"};
    bool        threw = false;
    try {
      cli.parse(a.argc(), a.data());
    }
    catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  // ── Error: option requiring value gets none ─────────────────────────
  {
    argv_holder a{"prog", "in.f4", "-l", "3x4x5", "-r"};  // -r at end, no value
    bool        threw = false;
    try {
      cli.parse(a.argc(), a.data());
    }
    catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  // ── Error: invalid integer ──────────────────────────────────────────
  {
    argv_holder a{"prog", "in.f4", "-l", "3x4x5", "-r", "not_a_number"};
    bool        threw = false;
    try {
      cli.parse(a.argc(), a.data());
    }
    catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  // ── Error: missing required (no positional, no dim3) ────────────────
  {
    // dim3 is required (no default) — omit -l to trigger
    argv_holder a{"prog", "in.f4"};
    bool        threw = false;
    try {
      cli.parse(a.argc(), a.data());
    }
    catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  // ── get<T>: type mismatch throws ────────────────────────────────────
  {
    argv_holder a{"prog", "in.f4", "-l", "3x4x5"};
    auto        r     = cli.parse(a.argc(), a.data());
    bool        threw = false;
    try {
      r.get<bool>("repeat");  // repeat is integer, not bool
    }
    catch (const std::runtime_error&) {
      threw = true;
    }
    assert(threw);
  }

  printf("test_arg_builder: PASS\n");
  return 0;
}
