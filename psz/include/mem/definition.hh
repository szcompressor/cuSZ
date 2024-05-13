#ifndef C7FE0EDE_832B_4A4D_9E43_85A84A0E025B
#define C7FE0EDE_832B_4A4D_9E43_85A84A0E025B

typedef enum pszmem_control {
  Malloc,
  MallocHost,
  MallocManaged,
  Free,
  FreeHost,
  FreeManaged,
  ClearHost,
  ClearDevice,
  H2D,
  H2H,
  D2H,
  D2D,
  ASYNC_H2D,
  ASYNC_H2H,
  ASYNC_D2H,
  ASYNC_D2D,
  ToFile,
  FromFile,
  ExtremaScan,
  DBG,
} pszmem_control;

#endif /* C7FE0EDE_832B_4A4D_9E43_85A84A0E025B */
