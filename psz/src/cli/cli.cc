#include "context.h"
#include "cli/cli.inl"
#include "port.hh"
#include "utils/query.hh"

int main(int argc, char** argv)
{
  auto ctx = pszctx_default_values();
  pszctx_create_from_argv(ctx, argc, argv);

  if (ctx->verbose) {
    CPU_QUERY;
    GPU_QUERY;
  }

  psz::CLI<float> psz_cli;
  psz_cli.dispatch(ctx);

  delete ctx;
}
