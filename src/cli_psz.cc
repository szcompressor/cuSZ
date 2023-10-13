#include "pipeline/cli.inl"
#include "port.hh"
#include "utils/query.hh"

int main(int argc, char** argv)
{
  auto ctx = new psz_context;
  pszctx_create_from_argv(ctx, argc, argv);

  if (ctx->verbose) {
    CPU_QUERY;
    GPU_QUERY;
  }
  
  cusz::CLI<float> cusz_cli;
  cusz_cli.dispatch(ctx);

  delete ctx;
}
