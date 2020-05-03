//
// Created by JianNan Tian on 2019-08-26.
//

#if not defined(TIMER_HH)
#define TIMER_HH

#include <chrono>

#include "__logging.hh"

using std::cerr;
using std::cout;
using std::endl;

using hires = std::chrono::high_resolution_clock;
typedef std::chrono::duration<double>                               duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> hires_clock_t;

typedef struct ClockCollection {
    hires_clock_t grand_commencement, grand_finale;
    hires_clock_t cxread_a, cpart_a, xaggreg_a, cpqr_a, xpr_a, cxhuffman_a, cxzstd_a, cxwrite_a;
    hires_clock_t cxread_z, cpart_z, xaggreg_z, cpqr_z, xpr_z, cxhuffman_z, cxzstd_z, cxwrite_z;
    duration_t    cxread_e, cpart_e, xaggreg_e, cpqr_e, xpr_e, cxhuffman_e, cxzstd_e, cxwrite_e;
    hires_clock_t cumalloc_a, cufree_a, cutrans_a, cumemh2d_a, cumemd2h_a;
    hires_clock_t cumalloc_z, cufree_z, cutrans_z, cumemh2d_z, cumemd2h_z;
duration_t grand_e2e;
    duration_t    cumalloc_e, cufree_e, cutrans_e, cumemh2d_e, cumemd2h_e;
    double        total_elapsed_io, total_elapsed_wo_lossless, total_elapsed_zstd, total_elapsed_huffman;

    void summarize(bool isCompressing, bool isCUDA = false) {
        cxread_e  = cxread_z - cxread_a;
        cxwrite_e = cxwrite_z - cxwrite_a;

        cxhuffman_e = cxhuffman_z - cxhuffman_a;
        cxzstd_e    = cxzstd_z - cxzstd_a;

        cpqr_e = cpqr_z - cpqr_a;
        xpr_e  = xpr_z - xpr_a;

        cpart_e   = cpart_z - cpart_a;
        xaggreg_e = xaggreg_z - xaggreg_a;

        cumalloc_e = cumalloc_z - cumalloc_a;
        cutrans_e  = cutrans_z - cutrans_a;
        cufree_e   = cufree_z - cufree_a;
        cumemh2d_e = cumemh2d_z - cumemh2d_a;
        cumemd2h_e = cumemd2h_z - cumemd2h_a;

grand_e2e = grand_finale - grand_commencement;
cout << "new e2e: " <<  grand_e2e.count() << endl;

        total_elapsed_io          = cxread_e.count() + cxwrite_e.count();
        total_elapsed_wo_lossless = 0 + cpart_e.count() + xaggreg_e.count()     //
                                    + cpqr_e.count() + xpr_e.count()            //
                                    + cutrans_e.count()                         //
                                    + cumalloc_e.count() + cufree_e.count()     //
                                    + cumemh2d_e.count() + cumemd2h_e.count();  //
        total_elapsed_huffman = total_elapsed_wo_lossless + cxhuffman_e.count();
        total_elapsed_zstd    = total_elapsed_huffman + cxzstd_e.count();

        //logging::timeElapsed(cxread_e.count(), "read (sec)");
        //logging::timeElapsed(cxwrite_e.count(), "write (sec)");
        cout << "------------------------------------------------------------" << endl;
        cout << "partial  \t" << std::setw(8) << "(sec)"
             << "\t% Huff-e2e.time\t% (Huff+Zstd)-e2e.time" << endl;
        // clang-format off
        if (isCompressing) {
            //logging::timeElapsed(cpart_e.count(),     total_elapsed_huffman, total_elapsed_zstd, "partition");
            logging::timeElapsed(cpqr_e.count(),      total_elapsed_huffman, total_elapsed_zstd, "KERNEL.c ");
            logging::timeElapsed(cxhuffman_e.count(), total_elapsed_huffman, total_elapsed_zstd, "HUFFMAN.c");
            logging::timeElapsed(cxzstd_e.count(),    total_elapsed_huffman, total_elapsed_zstd, "zstd.c   ");

        } else {
            logging::timeElapsed(cxhuffman_e.count(), total_elapsed_huffman, total_elapsed_zstd, "HUFFMAN.x");
            logging::timeElapsed(cxzstd_e.count(),    total_elapsed_huffman, total_elapsed_zstd, "zstd.x   ");
            logging::timeElapsed(xpr_e.count(),       total_elapsed_huffman, total_elapsed_zstd, "KERNEL.x ");
            //logging::timeElapsed(xaggreg_e.count(),   total_elapsed_huffman, total_elapsed_zstd, "aggregate");
        }

        if (isCUDA and isCompressing) {
        cout << "------------------------------------------------------------" << endl;
            logging::timeElapsed(cumalloc_e.count(), total_elapsed_huffman, total_elapsed_zstd, "cumalloc");
            logging::timeElapsed(cufree_e.count(),   total_elapsed_huffman, total_elapsed_zstd, "cufree  ");
            //logging::timeElapsed(cutrans_e.count(),  total_elapsed_huffman, total_elapsed_zstd, "cutrans ");
            logging::timeElapsed(cumemh2d_e.count(), total_elapsed_huffman, total_elapsed_zstd, "cumemh2d");
            logging::timeElapsed(cumemd2h_e.count(), total_elapsed_huffman, total_elapsed_zstd, "cumemd2h");
        }
        cout << "------------------------------------------------------------" << endl;
        logging::timeElapsed(total_elapsed_huffman, total_elapsed_huffman, total_elapsed_zstd, "Huff.e2e");
        logging::timeElapsed(total_elapsed_zstd,    total_elapsed_zstd,    total_elapsed_zstd, "Hf+Zstd.e2e");
        // clang-format on
        cout << "============================================================" << endl;
        //        cout << "throughput" << endl;
        //        logging::throughput(d->ori_size, timer->xpr_e.count(), timer->xpr_e.count(), "KERNEL");
        //        logging::throughput(d->ori_size, timer->total_elapsed_huffman, timer->total_elapsed_zstd, "end2end");
        //        cout << "============================================================" << endl;
    }

} clock_batch_t;

#endif  // TIMER_HH
