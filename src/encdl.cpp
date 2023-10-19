#include "encdl.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xtensor_forward.hpp>

void encDl(params_s *params) {

    // Input file
    std::ifstream in_file;

    // Read info bits
    in_file.open("./dl/tv0/info_bits.txt");
    xt::xarray<int> infoBits = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << "infoBits:" << std::endl << xt::transpose(infoBits) << std::endl;
    in_file.close();

    // Read CRC matrix
    in_file.open("./dl/tv0/crc_gen_m.txt");
    xt::xarray<int> crcGenVec = xt::ravel(xt::load_csv<short>(in_file));
    xt::xarray<int> crcGenMtx = xt::transpose(crcGenVec.reshape({params->P, params->K}));
    std::cout << "crcGenMtx:" << std::endl
              << xt::print_options::line_width(160) << crcGenMtx << std::endl;
    in_file.close();

    // CRC computation
    xt::xarray<int> crcBits = xt::linalg::dot(
        xt::concatenate(xt::xtuple(xt::ones<int>({params->P}), infoBits)), crcGenMtx);
    crcBits %= 2;
    std::cout << "crcBits:" << std::endl << xt::transpose(crcBits) << std::endl;

    // Read RNTI bits
    in_file.open("./dl/tv0/rnti_bits.txt");
    xt::xarray<int> rntiBits = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << "rntiBits:" << std::endl << xt::transpose(rntiBits) << std::endl;
    in_file.close();

    // CRC scramble
    xt::xarray<int> scrBits =
        crcBits ^ xt::concatenate(xt::xtuple(
                      xt::zeros<int>({params->P - rntiBits.size()}), rntiBits));
    std::cout << "scrBits:" << std::endl << xt::transpose(scrBits) << std::endl;

    // CRC attachment
    xt::xarray<int> infoCrcBits = xt::concatenate(xt::xtuple(infoBits, scrBits));
    std::cout << xt::print_options::line_width(160) << "infoCrcBits:" << std::endl
              << xt::transpose(infoCrcBits) << std::endl;

    // Read CRC interleaver pattern
    in_file.open("./dl/tv0/crc_interleaver_pattern.txt");
    xt::xarray<int> crcIntrl = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << xt::print_options::line_width(160) << "crcIntrl:" << std::endl
              << xt::transpose(crcIntrl) << std::endl;
    in_file.close();

    // CRC interleaver
    xt::xarray<int> intrlBits = xt::zeros_like(infoCrcBits);
    intrlBits = xt::index_view(infoCrcBits, crcIntrl);
    std::cout << xt::print_options::line_width(160) << "intrlBits:" << std::endl
              << xt::transpose(intrlBits) << std::endl;

    // Read CRC interleaver pattern
    in_file.open("./dl/tv0/info_bit_pattern.txt");
    xt::xarray<int> infoIntrl = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << xt::print_options::line_width(160) << "infoIntrl:" << std::endl
              << xt::transpose(infoIntrl) << std::endl;
    in_file.close();

    // Frozen bit insertion
    xt::xarray<int> frozenBits = xt::zeros<int>({params->N});
    xt::filter(frozenBits, infoIntrl > 0) = intrlBits;
    std::cout << xt::print_options::line_width(160) << "frozenBits:" << std::endl
              << xt::transpose(frozenBits) << std::endl;

    // Read encoder matrix
    in_file.open("./dl/tv0/enc_gen_m.txt");
    xt::xarray<int> encGenVec = xt::ravel(xt::load_csv<short>(in_file));
    xt::xarray<int> encGenMtx = xt::transpose(encGenVec.reshape({params->N, params->N}));
    std::cout << "encGenMtx:" << std::endl
              << xt::print_options::line_width(160) << xt::print_options::edge_items(20)
              << encGenMtx << std::endl;
    in_file.close();

    // Encoding
    xt::xarray<int> encBits = xt::linalg::dot(frozenBits, encGenMtx);
    encBits %= 2;
    std::cout << xt::print_options::line_width(160) << "encBits:" << std::endl
              << xt::transpose(encBits) << std::endl;

    // Read rate matching pattern
    in_file.open("./dl/tv0/rate_matching_pattern.txt");
    xt::xarray<int> encIntrl = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << xt::print_options::line_width(160) << "encIntrl:" << std::endl
              << xt::transpose(encIntrl) << std::endl;
    in_file.close();

    // Rate matching
    xt::xarray<int> rmBits = xt::zeros<int>({params->E});
    rmBits = xt::index_view(encBits, encIntrl);
    std::cout << xt::print_options::line_width(160) << "rmBits:" << std::endl
              << xt::transpose(rmBits) << std::endl;

    // Read rate matched reference bits
    in_file.open("./dl/tv0/rm_bits.txt");
    xt::xarray<int> rmRefs = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << xt::print_options::line_width(160) << "rmRefs:" << std::endl
              << xt::transpose(rmRefs) << std::endl;
    in_file.close();

    // Check results
    xt::xarray<int> checkBits = xt::abs(rmRefs - rmBits);
    int nDiffBits = xt::sum(checkBits)();
    std::cout << "nDiffBits: " << nDiffBits << std::endl;
}