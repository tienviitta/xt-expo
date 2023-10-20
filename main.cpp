#include "encdl.h"
#include "ex1.h"
#include "util.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    // CLI args
    for (int i = 1; i < argc; ++i)
        std::cout << argv[i] << "\n";
    fs::path paramsPath{argv[1]}; // Note! Testcase path as argv[1]!

    // Read params
    params_s params;
    readParams(paramsPath, &params);

    // Encoding
    encDl(paramsPath, &params);

    // ex1_run();
    // ex2_run();
    // ex3_run();
    // ex4_run();

    // ex1_vec_run();
    // ex2_vec_run();
    // ex3_vec_run();
    // ex4_vec_run();

    // ex1_view_run();
    ex2_ind_run();
    ex3_ind_run();
    // ex1_ind_run();
    // ex1_rnd_run();

    // ex1_csv_run();

    // ex1_qck_run();
    // ex1_red_run();
    // ex1_man_run();

    // ex1_crc_run();
    // ex2_crc_run();
    // ex1_mpow_run();

    // ex1_cmplx_run();
    // ex2_cmplx_run();

    return 0;
}
