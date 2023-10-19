#include "util.h"
#include "encdl.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>

void readParams(const char *fn, params_s *params) {
    std::ifstream in_file;
    in_file.open(fn);
    xt::xarray<int> params_f = xt::ravel(xt::load_csv<int>(in_file));
    std::cout << "params_f:" << std::endl << xt::transpose(params_f) << std::endl;
    for (int i = 0; i < params_f.size(); ++i) {
        switch (i) {
        case 0:
            params->A = params_f.at(i);
            break;
        case 1:
            params->P = params_f.at(i);
            break;
        case 2:
            params->K = params_f.at(i);
            break;
        case 3:
            params->E = params_f.at(i);
            break;
        case 4:
            params->N = params_f.at(i);
            break;
        default:
            break;
        }
    }
    in_file.close();
}