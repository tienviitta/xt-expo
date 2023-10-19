#ifndef ENCDL_H_
#define ENCDL_H_

#include <filesystem>

namespace fs = std::filesystem;

typedef struct params_s {
    int A;
    int P;
    int K;
    int E;
    int N;
} params_s;

void encDl(fs::path path, params_s *params);

#endif // ENCDL_H_