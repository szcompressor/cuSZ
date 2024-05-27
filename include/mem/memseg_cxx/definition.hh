#ifndef C7FE0EDE_832B_4A4D_9E43_85A84A0E025B
#define C7FE0EDE_832B_4A4D_9E43_85A84A0E025B

enum pszmem_control_stream {
  Malloc,
  MallocHost,
  MallocManaged,
  Free,
  FreeHost,
  FreeManaged,
  ClearHost,
  ClearDevice,
  H2D,
  D2H,
  ASYNC_H2D,
  ASYNC_D2H,
  ToFile,
  FromFile,
  ExtremaScan,
};

using pszmem_control = pszmem_control_stream;

#endif /* C7FE0EDE_832B_4A4D_9E43_85A84A0E025B */
