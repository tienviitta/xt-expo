#include "encdl.h"
#include "util.h"
#include <iostream>

int main(int argc, char *argv[]) {

    // Read params
    params_s params;
    const char params_fn[] = "./dl/tv0/params.txt";
    readParams(params_fn, &params);

    // Encoding
    encDl(&params);

    // ex1_run();
    // ex2_run();
    // ex3_run();
    // ex4_run();

    // ex1_vec_run();
    // ex2_vec_run();
    // ex3_vec_run();
    // ex4_vec_run();

    // ex1_view_run();
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
