//
// Created by JianNan Tian on 10/3/19.
//

#include "parser.hh"

#include <boost/program_options.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "types.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::vector;
namespace po = boost::program_options;

Context::Context() {}

void Context::debug() {
    if (D == nullptr or C == nullptr) {
        cerr << "D or C is not initialized!" << endl;
    } else {
        D->debug();
        cout << endl;
        C->debug();
    }
}

/**
 *
 * @param ac    argument count
 * @param av    argument values
 * @param ctx   context
 * @param desc  program option description
 * @return      status
 */
int parser(int ac, char** av, ctx_t* ctx, po::options_description& desc) {
    size_t capacity;
    int    eb_base10_exp, eb_base10_abs;

    try {
        /* ---------------------------------------------------------------------------------------------------- */
        desc.add_options()                                                                                                                    //
            ("help,h", "print help message")                                                                                                  //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("input-file,i", po::value<std::string>(&(ctx->input_file))->required(), "\33[1;31;40m required \33[0m specify the input file")(  //
                "data-type,t", po::value<std::string>(&(ctx->data_type))->required(), "\33[1;31;40m required \33[0m specify the data type")   //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("compress,c", po::bool_switch()->default_value(false), "to compress a file")                                                     //
            ("decompress,x", po::bool_switch()->default_value(false), "to decompress a file")                                                 //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("dims,d", po::value<std::vector<size_t> >()->multitoken()->required(),                                                           //
             "\33[1;31;40m required \33[0m specify dimensions, e.g., \"1800 3600\" for \"...-1800x3600.f32\" dataset")                        //
            ("sub-dims,s", po::value<std::vector<size_t> >()->multitoken(),                                                                   //
             "specify block dimensions (sub-dims) in accordance to the dataset dimensions")                                                   //
            ("overwrite,o", po::bool_switch()->default_value(false), "enable file overwrite")                                                 //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("abs,a", po::bool_switch()->default_value(true), "absolute mode")                                                                //
            ("rel,r", po::bool_switch()->default_value(false), "relative mode")                                                               //
            ("point-wise,p", po::bool_switch()->default_value(false), "pointwise mode")                                                       //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("cap,C", po::value<size_t>()->default_value(32768), "quantization bin capacity")                                                 //
            ("eb,e", po::value<int>()->default_value(-3), "error bound in base-10 exponent, e.g., \"-3\"")                                    //
            ("bin-eb", po::value<int>(), "error bound in base-2 exponent, e.g., \"-10\"")                                                     //
            ("abs-eb", po::value<double>(), "error bound in base-10 absulute number, e.g., \"15\"")                                           //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("native-seq", po::bool_switch()->default_value(true), "native CPU run")                                                          //
            ("native-cuda,u", po::bool_switch(), "offloading to GPU, ruuning with native CUDA")                                               //
            ("kokkos-seq", po::bool_switch(), "(Kokkos) CPU sequantial run")                                                                  //
            ("kokkos-omp", po::bool_switch(), "(Kokkos) omp parallelization")                                                                 //
            ("kokkos-cuda", po::bool_switch(), "(Kokkos) offloading to GPU")                                                                  //
            ("raja-seq", po::bool_switch(), "(RAJA) CPU sequantial run")                                                                      //
            ("raja-simd", po::bool_switch(), "(RAJA) compiler hinted SIMD")                                                                   //
            ("raja-omp", po::bool_switch(), "(RAJA) omp parallelization")                                                                     //
            ("raja-omp-gpu", po::bool_switch(), "(RAJA) omp-gpu parallelization")                                                             //
            ("raja-cuda", po::bool_switch(), "(RAJA) offloading to GPU")                                                                      //
            ;                                                                                                                                 //
        desc.add_options()                                                                                                                    //
            ("print-metadata,P", po::bool_switch()->default_value(true), "print metadata")                                                    //
            ("stat", po::bool_switch()->default_value(true), "get statistics by comparing with the orignal dataset")                          //
            ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }
        po::notify(vm); /* raise error if the required arguments are not there */

        // cout << vm["compress"].as<bool>() << endl;
        // cout << vm["decompress"].as<bool>() << endl;
        bool is_SZc = vm["compress"].as<bool>();
        bool is_SZx = vm["decompress"].as<bool>();
        if ((is_SZc and is_SZx) or ((not is_SZc) and (not is_SZx))) {
            throw std::runtime_error("must exclusively choose compression or decompression!");
        }

        /* filename and detatype */
        ctx->data_type  = vm["data-type"].as<std::string>();
        ctx->input_file = vm["input-file"].as<std::string>();

        vector<size_t> dims, segSizes;
        size_t         ndims;

        /* parsing dimensions and subdimensions */
        dims    = vm["dims"].as<vector<size_t> >();
        segSizes = vm["sub-dims"].as<vector<size_t> >();
        ndims  = dims.size();
        if (dims.size() != segSizes.size()) throw std::runtime_error("The number data dimensions must match that of sub dimensions!");
        while (dims.size() != maxDim) dims.push_back(1U);
        while (segSizes.size() != maxDim) segSizes.push_back(1U);

        ctx->D = new dim_t(ndims, dims, segSizes, 1);
        // if (vm["print-metadata"].as<bool>()) D->debug();

        /* get subdims (block sizes) */
        if (not vm["parts"].empty()) {
            segSizes = vm["parts"].as<vector<size_t> >();
            if (segSizes.size() != dims.size()) throw std::runtime_error("dims and segSizes do now have the same length!");
            while (segSizes.size() != maxDim) segSizes.push_back(1U);
        } else {
            //
        }

        /* configurations */
        capacity = vm["cap"].as<size_t>();
        if (vm.count("abs-eb")) { /* use absolute value as error bound*/
            if (not vm.count("abs")) throw std::runtime_error("must use abs mode when overriding the error bound!");
            ctx->C = new config_t(capacity, vm["abs-eb"].as<double>());
        } else if (vm.count("bin-eb")) { /* do dec2bin conversion */
            ctx->C = new config_t(0, vm["bin-eb"].as<int>(), capacity);
        } else { /* "eb" has default value */
            cout << "No specifying error bound, use " << vm["eb"].as<int>() << " (default value)!" << endl;
            ctx->C = new config_t(vm["eb"].as<int>(), 0, capacity);
        }
        /* ---------------------------------------------------------------------------------------------------- */
    } catch (const boost::program_options::required_option& e) {
        cerr << "\33[1;33;40m error \33[0m " << e.what() << "\n\n";
        std::cout << desc << std::endl;
        return 1;
    } catch (exception& e) {
        cerr << "error: " << e.what() << "\n";
        std::cout << desc << std::endl;
        return 1;
    } catch (...) {
        cerr << "Exception of unknown type!\n";
        return 1;
    }
    return 0;
}

//#if defined(TEST_INPLACE)
 int main(int ac, char** av) {
    po::options_description desc("listing options");

    auto* ctx = new ctx_t();
    if (int _status = parser(ac, av, ctx, desc) != 0) {
        cerr << "parser not working correctly!" << endl;  // TODO change to throw
    }
    ctx->debug();

    return 0;
}
//#endif
