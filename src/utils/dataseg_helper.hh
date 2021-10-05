/**
 * @file dataseg_helper.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_UTILS_DATASEG_HELPER_HH
#define CUSZ_UTILS_DATASEG_HELPER_HH

#include "../common/definition.hh"
#include "../header.hh"
#include "cuda_mem.cuh"

#include <unordered_map>
#include <vector>

struct DatasegHelper {
    using BYTE = uint8_t;

    static void header_nbyte_from_dataseg(cuszHEADER* header, DataSeg& dataseg)
    {
        header->nbyte.book           = dataseg.nbyte.at(cuszSEG::BOOK);
        header->nbyte.revbook        = dataseg.nbyte.at(cuszSEG::REVBOOK);
        header->nbyte.outlier        = dataseg.nbyte.at(cuszSEG::OUTLIER);
        header->nbyte.huff_meta      = dataseg.nbyte.at(cuszSEG::HUFF_META);
        header->nbyte.huff_bitstream = dataseg.nbyte.at(cuszSEG::HUFF_DATA);
    }

    static void dataseg_nbyte_from_header(cuszHEADER* header, DataSeg& dataseg)
    {
        dataseg.nbyte.at(cuszSEG::BOOK)      = header->nbyte.book;
        dataseg.nbyte.at(cuszSEG::REVBOOK)   = header->nbyte.revbook;
        dataseg.nbyte.at(cuszSEG::OUTLIER)   = header->nbyte.outlier;
        dataseg.nbyte.at(cuszSEG::HUFF_META) = header->nbyte.huff_meta;
        dataseg.nbyte.at(cuszSEG::HUFF_DATA) = header->nbyte.huff_bitstream;
    }

    static void compress_time_conslidate_report(DataSeg& dataseg, std::vector<uint32_t>& offsets)
    {
        ReportHelper::print_datasegment_tablehead();

        // print long numbers with thousand separator
        // https://stackoverflow.com/a/7455282
        // https://stackoverflow.com/a/11695246
        setlocale(LC_ALL, "");

        for (auto i = 0; i < 8; i++) {
            const auto& name       = dataseg.order2name.at(i);
            auto        this_nbyte = dataseg.nbyte.at(name);

            auto o = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(this_nbyte);
            offsets.push_back(o);

            if (this_nbyte != 0)
                printf(
                    "  %-18s\t%'12lu\t%'15lu\t%'15lu\n",  //
                    dataseg.get_namestr(name).c_str(),    //
                    (size_t)this_nbyte,                   //
                    (size_t)offsets.at(i),                //
                    (size_t)offsets.back());
        }
    }
};

#endif