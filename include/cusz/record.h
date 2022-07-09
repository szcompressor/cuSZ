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

struct cusz_record_entry;

struct cusz_record_entry {
    const char* name;
    double      time;

    struct cusz_record_entry* next;
};

typedef struct cusz_record {
    int n;

    struct cusz_record_entry* head;
} cusz_record;

#ifdef __cplusplus
}
#endif

#endif
