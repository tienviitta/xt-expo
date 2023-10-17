cmake_minimum_required(VERSION 3.16)
project(xt_ex VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

include(CTest)
enable_testing()

add_executable(xt_ex main.cpp src/ex1.cpp src/ex1.h)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

target_include_directories(xt_ex PRIVATE src)
target_link_libraries(xt_ex xtensor xtensor::optimize xtensor::use_xsimd)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
