#ifndef D4543D36_B236_4E84_B7C9_AADFE5E84628
#define D4543D36_B236_4E84_B7C9_AADFE5E84628

#include <stddef.h>
#include <stdint.h>

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
      uint32_t entry[FZGHEADER_END];
    };
  };
} fzg_header;

#endif /* D4543D36_B236_4E84_B7C9_AADFE5E84628 */
