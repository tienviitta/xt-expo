#ifndef UTIL_H_
#define UTIL_H_

#include "encdl.h"
#include <filesystem>

namespace fs = std::filesystem;

void readParams(fs::path path, params_s *params);

#endif // UTIL_H_
