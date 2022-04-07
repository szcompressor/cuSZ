/**
 * @file timerecord_viewer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-09
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CLI_TIMERECORD_VIEWER_HH
#define CLI_TIMERECORD_VIEWER_HH

#include "../common/types.hh"

namespace cusz {

struct TimeRecordViewer {
    static float get_throughput(float milliseconds, size_t bytes)
    {
        auto GiB     = 1.0 * 1024 * 1024 * 1024;
        auto seconds = milliseconds * 1e-3;
        return bytes / GiB / seconds;
    }

    static void println_throughput(const char* s, float timer, size_t bytes)
    {
        if (timer == 0.0) return;

        auto t = get_throughput(timer, bytes);
        printf("  %-12s %'12f %'10.2f\n", s, timer, t);
    };

    static void println_throughput_tablehead()
    {
        printf(
            "\n  \e[1m\e[31m%-12s %12s %10s\e[0m\n",  //
            const_cast<char*>("kernel"),              //
            const_cast<char*>("time, ms"),            //
            const_cast<char*>("GiB/s")                //
        );
    }

    static double get_total_time(timerecord_t r)
    {
        double total = 0.0;
        std::for_each(r->begin(), r->end(), [&](TimeRecordTuple t) { return total += std::get<1>(t); });
        return total;
    }
    static void view_compression(timerecord_t r, size_t bytes, size_t compressed_bytes = 0)
    {
        auto report_cr = [&]() {
            auto cr = 1.0 * bytes / compressed_bytes;
            if (compressed_bytes != 0) printf("  %-*s %.2f\n", 20, "compression ratio", cr);
        };

        TimeRecord reflow;

        {  // reflow
            TimeRecordTuple book_tuple;

            auto total_time    = get_total_time(r);
            auto subtotal_time = total_time;

            for (auto& i : *r) {
                auto item = std::string(std::get<0>(i));
                if (item == "book") {
                    book_tuple = i;
                    subtotal_time -= std::get<1>(i);
                }
                else {
                    reflow.push_back(i);
                }
            }
            reflow.push_back({const_cast<const char*>("(subtotal)"), subtotal_time});
            printf("\e[2m");
            reflow.push_back(book_tuple);
            reflow.push_back({const_cast<const char*>("(total)"), total_time});
            printf("\e[0m");
        }

        printf("\n(c) COMPRESSION REPORT\n");
        report_cr();

        ReportHelper::println_throughput_tablehead();
        for (auto& i : reflow) ReportHelper::println_throughput(std::get<0>(i), std::get<1>(i), bytes);

        printf("\n");
    }

    static void view_decompression(timerecord_t r, size_t bytes)
    {
        printf("\n(d) deCOMPRESSION REPORT\n");

        auto total_time = get_total_time(r);
        (*r).push_back({const_cast<const char*>("(total)"), total_time});

        ReportHelper::println_throughput_tablehead();
        for (auto& i : *r) ReportHelper::println_throughput(std::get<0>(i), std::get<1>(i), bytes);

        printf("\n");
    }
};

}  // namespace cusz

#endif
