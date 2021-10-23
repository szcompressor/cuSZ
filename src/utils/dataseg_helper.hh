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

// #include "../common/definition.hh"
#include "../header.hh"
#include "cuda_mem.cuh"

#include <unordered_map>
#include <vector>

class DataSeg {
   public:
    std::unordered_map<cuszSEG, int> name2order = {
        {cuszSEG::HEADER, 0}, {cuszSEG::BOOK, 1},      {cuszSEG::QUANT, 2},     {cuszSEG::REVBOOK, 3},
        {cuszSEG::SPFMT, 4},  {cuszSEG::HUFF_META, 5}, {cuszSEG::HUFF_DATA, 6},  //
        {cuszSEG::ANCHOR, 7}};

    std::unordered_map<int, cuszSEG> order2name = {
        {0, cuszSEG::HEADER}, {1, cuszSEG::BOOK},      {2, cuszSEG::QUANT},     {3, cuszSEG::REVBOOK},
        {4, cuszSEG::SPFMT},  {5, cuszSEG::HUFF_META}, {6, cuszSEG::HUFF_DATA},  //
        {7, cuszSEG::ANCHOR}};

    std::unordered_map<cuszSEG, uint32_t> nbyte = {
        {cuszSEG::HEADER, sizeof(cusz_header)},
        {cuszSEG::BOOK, 0U},
        {cuszSEG::QUANT, 0U},
        {cuszSEG::REVBOOK, 0U},
        {cuszSEG::ANCHOR, 0U},
        {cuszSEG::SPFMT, 0U},
        {cuszSEG::HUFF_META, 0U},
        {cuszSEG::HUFF_DATA, 0U}};

    std::unordered_map<cuszSEG, std::string> name2str{
        {cuszSEG::HEADER, "HEADER"},       {cuszSEG::BOOK, "BOOK"},          {cuszSEG::QUANT, "QUANT"},
        {cuszSEG::REVBOOK, "REVBOOK"},     {cuszSEG::ANCHOR, "ANCHOR"},      {cuszSEG::SPFMT, "SPFMT"},
        {cuszSEG::HUFF_META, "HUFF_META"}, {cuszSEG::HUFF_DATA, "HUFF_DATA"}};

    std::vector<uint32_t> offset;

    uint32_t    get_offset(cuszSEG name) { return offset.at(name2order.at(name)); }
    std::string get_namestr(cuszSEG name) { return name2str.at(name); }
};

struct DatasegHelper {
    using BYTE = uint8_t;

    static void header_nbyte_from_dataseg(cuszHEADER* header, DataSeg& dataseg)
    {
        header->nbyte.book      = dataseg.nbyte.at(cuszSEG::BOOK);
        header->nbyte.revbook   = dataseg.nbyte.at(cuszSEG::REVBOOK);
        header->nbyte.spfmt     = dataseg.nbyte.at(cuszSEG::SPFMT);
        header->nbyte.huff_meta = dataseg.nbyte.at(cuszSEG::HUFF_META);
        header->nbyte.huff_data = dataseg.nbyte.at(cuszSEG::HUFF_DATA);
        header->nbyte.anchor    = dataseg.nbyte.at(cuszSEG::ANCHOR);
    }

    static void dataseg_nbyte_from_header(cuszHEADER* header, DataSeg& dataseg)
    {
        dataseg.nbyte.at(cuszSEG::BOOK)      = header->nbyte.book;
        dataseg.nbyte.at(cuszSEG::REVBOOK)   = header->nbyte.revbook;
        dataseg.nbyte.at(cuszSEG::SPFMT)     = header->nbyte.spfmt;
        dataseg.nbyte.at(cuszSEG::HUFF_META) = header->nbyte.huff_meta;
        dataseg.nbyte.at(cuszSEG::HUFF_DATA) = header->nbyte.huff_data;
        dataseg.nbyte.at(cuszSEG::ANCHOR)    = header->nbyte.anchor;
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