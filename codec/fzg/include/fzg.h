#ifndef FZG_H
#define FZG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef void* fzg_stream_t;

#define FZGHEADER_HEADER 0
#define FZGHEADER_BITFLAG 1
#define FZGHEADER_START_POS 2
#define FZGHEADER_BITSTREAM 3
#define FZGHEADER_END 4

#define TODO_CHANGE_FZGHEADER_SIZE 128

typedef struct fzg_header {
  union {
    struct {
      uint8_t __[TODO_CHANGE_FZGHEADER_SIZE];
    };

    struct {
      size_t original_len;
      uint32_t entry[FZGHEADER_END + 1];
    };
  };
} fzg_header;

#ifdef __cplusplus
}
#endif

#endif
