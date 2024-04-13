/**
 * @file record.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-30
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_RECORD_H
#define CUSZ_RECORD_H

#ifdef __cplusplus
extern "C" {
#endif

struct psz_record_entry;

struct psz_record_entry {
  const char* name;
  double time;

  struct psz_record_entry* next;
};

typedef struct psz_record {
  int n;

  struct psz_record_entry* head;
} psz_record;

#ifdef __cplusplus
}
#endif

#endif
