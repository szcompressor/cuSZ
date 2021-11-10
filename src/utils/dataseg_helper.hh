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
    std::unordered_map<cusz::SEG, int> name2order = {
        {cusz::SEG::HEADER, 0}, {cusz::SEG::BOOK, 1},      {cusz::SEG::QUANT, 2},     {cusz::SEG::REVBOOK, 3},
        {cusz::SEG::SPFMT, 4},  {cusz::SEG::HUFF_META, 5}, {cusz::SEG::HUFF_DATA, 6},  //
        {cusz::SEG::ANCHOR, 7}};

    std::unordered_map<int, cusz::SEG> order2name = {
        {0, cusz::SEG::HEADER}, {1, cusz::SEG::BOOK},      {2, cusz::SEG::QUANT},     {3, cusz::SEG::REVBOOK},
        {4, cusz::SEG::SPFMT},  {5, cusz::SEG::HUFF_META}, {6, cusz::SEG::HUFF_DATA},  //
        {7, cusz::SEG::ANCHOR}};

    std::unordered_map<cusz::SEG, uint32_t> nbyte = {
        {cusz::SEG::HEADER, sizeof(cuszHEADER)},
        {cusz::SEG::BOOK, 0U},
        {cusz::SEG::QUANT, 0U},
        {cusz::SEG::REVBOOK, 0U},
        {cusz::SEG::ANCHOR, 0U},
        {cusz::SEG::SPFMT, 0U},
        {cusz::SEG::HUFF_META, 0U},
        {cusz::SEG::HUFF_DATA, 0U}};

    std::unordered_map<cusz::SEG, std::string> name2str{
        {cusz::SEG::HEADER, "HEADER"},       {cusz::SEG::BOOK, "BOOK"},          {cusz::SEG::QUANT, "QUANT"},
        {cusz::SEG::REVBOOK, "REVBOOK"},     {cusz::SEG::ANCHOR, "ANCHOR"},      {cusz::SEG::SPFMT, "SPFMT"},
        {cusz::SEG::HUFF_META, "HUFF_META"}, {cusz::SEG::HUFF_DATA, "HUFF_DATA"}};

    std::vector<uint32_t> offset;

    uint32_t    get_offset(cusz::SEG name) { return offset.at(name2order.at(name)); }
    std::string get_namestr(cusz::SEG name) { return name2str.at(name); }
};

struct DatasegHelper {
    using BYTE = uint8_t;

    static void header_nbyte_from_dataseg(cuszHEADER* header, DataSeg& dataseg)
    {
        header->nbyte.book      = dataseg.nbyte.at(cusz::SEG::BOOK);
        header->nbyte.revbook   = dataseg.nbyte.at(cusz::SEG::REVBOOK);
        header->nbyte.spfmt     = dataseg.nbyte.at(cusz::SEG::SPFMT);
        header->nbyte.huff_meta = dataseg.nbyte.at(cusz::SEG::HUFF_META);
        header->nbyte.huff_data = dataseg.nbyte.at(cusz::SEG::HUFF_DATA);
        header->nbyte.anchor    = dataseg.nbyte.at(cusz::SEG::ANCHOR);
    }

    static void dataseg_nbyte_from_header(cuszHEADER* header, DataSeg& dataseg)
    {
        dataseg.nbyte.at(cusz::SEG::BOOK)      = header->nbyte.book;
        dataseg.nbyte.at(cusz::SEG::REVBOOK)   = header->nbyte.revbook;
        dataseg.nbyte.at(cusz::SEG::SPFMT)     = header->nbyte.spfmt;
        dataseg.nbyte.at(cusz::SEG::HUFF_META) = header->nbyte.huff_meta;
        dataseg.nbyte.at(cusz::SEG::HUFF_DATA) = header->nbyte.huff_data;
        dataseg.nbyte.at(cusz::SEG::ANCHOR)    = header->nbyte.anchor;
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
