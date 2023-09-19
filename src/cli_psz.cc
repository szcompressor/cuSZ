#include "pipeline/cli.inl"
#include "port.hh"

int main(int argc, char** argv)
{
  // auto ctx = new cusz_context(argc, argv);
  auto ctx = new cusz_context;
  pszctx_create_from_argv(ctx, argc, argv);

  if (ctx->verbose) {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    Diagnostics::GetMachineProperties();
    GpuDiagnostics::GetDeviceProperty();
#elif defined(PSZ_USE_1API)
    printf("[psz::log::1api] machine info print is disabled temporarily.");
#endif
  }

  cusz::CLI<float> cusz_cli;
  cusz_cli.dispatch(ctx);
}
