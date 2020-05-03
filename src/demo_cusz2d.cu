// 200327

#include <algorithm>
#include <string>
#include <vector>

#include "SDRB.hh"
#include "__io.hh"
//#include "cusz_workflow.cuh"
#include "demo_cusz2d_workflow.cuh"
#include "types.hh"

namespace fm = cuSZ::FineMassive;
using std::string;
using std::vector;

#if defined(_q128)
const size_t DICT_SIZE = 128;
#elif defined(_q256)
const size_t DICT_SIZE = 256;
#elif defined(_q512)
const size_t DICT_SIZE = 512;
#elif defined(_q1024)
const size_t DICT_SIZE = 1024;
#elif defined(_q2048)
const size_t DICT_SIZE = 2048;
#elif defined(_q4096)
const size_t DICT_SIZE = 4096;
#elif defined(_q8192)
const size_t DICT_SIZE = 8192;
#elif defined(_q16384)
const size_t DICT_SIZE = 16384;
#elif defined(_q32768)
const size_t DICT_SIZE = 32768;
#else
const size_t DICT_SIZE = 65536;
#endif

const size_t BLK = 16;

int main(int argc, char** argv) {
    std::string eb_mode, dataset, datum_name;
    double      mantissa, exponent;
    size_t      new_y, new_x;

    size_t* dims_L16 = nullptr;

    cout << "******************************************* cuSZ configuration *******************************************" << endl;
    cout << "\e[46mThis program is specifically working for 2D cases, using " << DICT_SIZE << " quantization bins.\e[0m" << endl;
    if (argc < 6) {
    PRINT_HELP:
        cout << "\t./<program> <abs|rel2range> <mantissa> <exponent> <dataset> <datum path>" << endl;
        cout << "\t./democusz2d rel2range 1 -3 exafeldemo ./exafel-sample-raw-f32-59200x388.dat" << endl;
        cout << endl;
        cout << "Alternatively, to specify dimension BEFORE BINNING:" << endl;
        cout << "Currently, binning only supports even dimensions." << endl;
        cout << "\t./<program> <abs|rel2range> <mantissa> <exponent> <dataset> <datum path> <highest, y dim> <lowest, x dim>" << endl;
        cout << "\t./democusz2d rel2range 1 -3 exafeldemo /path/to/exafel_demo.dat 59200 388" << endl;
        exit(0);
    } else if (argc == 6) {
        eb_mode    = std::string(argv[1]);
        mantissa   = std::stod(argv[2]);
        exponent   = std::stod(argv[3]);
        dataset    = std::string(argv[4]);
        datum_name = std::string(argv[5]);

        dims_L16 = InitializeDims<BLK>(dataset, DICT_SIZE);
    } else if (argc == 8) {
        eb_mode    = std::string(argv[1]);
        mantissa   = std::stod(argv[2]);
        exponent   = std::stod(argv[3]);
        dataset    = std::string(argv[4]);
        datum_name = std::string(argv[5]);
        new_y      = (size_t)atoi(argv[6]);
        new_x      = (size_t)atoi(argv[7]);

        dims_L16 = InitializeDims<BLK>(dataset, DICT_SIZE, true, new_x, new_y);
    } else {
        goto PRINT_HELP;
    }

    cout << "filename:\t" << datum_name << endl;

    auto eb_config = new config_t(1024, mantissa, exponent);
    //    if (eb_mode.compare("rel2range") == 0) {
    if (eb_mode == "rel2range") {
        double value_range = GetDatumValueRange<float>(datum_name, dims_L16[LEN]);
        eb_config->ChangeToRelativeMode(value_range);
    }
    eb_config->debug();
    auto   ebs_L4 = InitializeErrorBoundFamily(eb_config);
    size_t num_outlier    = 0;

    if (DICT_SIZE > 256 and DICT_SIZE <= 65536) {
        fm::demo_c<float, uint16_t, BLK>(datum_name, dims_L16, ebs_L4, num_outlier);
        fm::demo_x<float, uint16_t, BLK>(datum_name, dims_L16, ebs_L4, num_outlier);
    } else {
        fm::demo_c<float, uint8_t, BLK>(datum_name, dims_L16, ebs_L4, num_outlier);
        fm::demo_x<float, uint8_t, BLK>(datum_name, dims_L16, ebs_L4, num_outlier);
    }
}
