#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <xtensor/xview.hpp>

namespace fs = std::filesystem;

/** xtensor

Usage:

Three container classes implementing multidimensional arrays are provided: xt::xarray and
xt::xtensor and xt::xtensor_fixed.
- xt::xarray can be reshaped dynamically to any number of dimensions. It is the container
that is the most similar to NumPy arrays.
- xt::xtensor has a dimension set at compilation time, which enables many optimizations.
For example, shapes and strides of xt::xtensor instances are allocated on the stack
instead of the heap.
- xt::xtensor_fixed has a shape fixed at compile time. This allows even more
optimizations, such as allocating the storage for the container on the stack, as well as
computing strides and backstrides at compile time, making the allocation of this container
extremely cheap.

The dynamic dimensionality of xt::xarray comes at a cost. Since the dimension is unknown
at build time, the sequences holding shape and strides of xt::xarray instances are
heap-allocated, which makes it significantly more expensive than xt::xtensor. Shape and
strides of xt::xtensor are stack-allocated which makes them more efficient.

xtensor provides overloads of traditional arithmetic operators for xt::xexpression
objects. All these operators are element-wise operators and apply the lazy broadcasting
rules explained in a previous section. See the API reference for a comprehensive list of
available functions. Like operators, the mathematical functions are element-wise functions
and apply the lazy broadcasting rules.

xtensor provides utilities to vectorize any scalar function (taking multiple scalar
arguments) into a function that will perform on xt::xexpression s, applying the lazy
broadcasting rules which we described in a previous section. These functions are called
xt::xfunction s. They are xtensor’s counterpart to numpy’s universal functions.

Views are used to adapt the shape of an xt::xexpression without changing it, nor copying
it. Views are convenient tools for assigning parts of an expression: since they do not
copy the underlying expression, assigning to the view actually assigns to the underlying
expression. xtensor provides many kinds of views.

Slices can be specified in the following ways:
- selection in a dimension by specifying an index (unsigned integer)
- xt::range(min, max), a slice representing the interval [min, max)
- xt::range(min, max, step), a slice representing the stepped interval [min, max)
- xt::all(), a slice representing all the elements of a dimension
- xt::newaxis(), a slice representing an additional dimension of length one
- xt::keep(i0, i1, i2, ...) a slice selecting non-contiguous indices to keep on the
underlying expression
- xt::drop(i0, i1, i2, ...) a slice selecting non-contiguous indices to drop on the
underlying expression

xt::xview does not perform a copy of the underlying expression. This means if you modify
an element of the xt::xview, you are actually also altering the underlying expression. In
the case of a tensor containing complex numbers, xtensor provides views returning
xt::xexpression corresponding to the real and imaginary parts of the complex numbers. Like
for other views, the elements of the underlying xt::xexpression are not copied. Functions
xt::real() and xt::imag() respectively return views on the real and imaginary part of a
complex expression. The returned value is an expression holding a closure on the passed
argument.

xtensor uses a lazy generator for random numbers. You need to assign them or use
xt::eval() to keep the generated values consistent.

xtensor can be configured via macros which must be defined before including any of its
headers. This can be achieved the following ways:
- either define them in the CMakeLists of your project, with target_compile_definitions
cmake command.
- or create a header where you define all the macros you want and then include the headers
you need. Then include this header whenever you need xtensor in your project.

Basics:

Tensor types
- xarray<T>: tensor that can be reshaped to any number of dimensions.
- xtensor<T, N>: tensor with a number of dimensions set to N at compile time.
- xtensor_fixed<T, xshape<I, J, K>: tensor whose shape is fixed at compile time.
- xchunked_array<CS>: chunked array using the CS chunk storage.



*/

void ex1_run() {
    xt::xarray<double> arr1{{1.0, 2.0, 3.0}, {2.0, 5.0, 7.0}, {2.0, 5.0, 7.0}};
    xt::xarray<double> arr2{5.0, 6.0, 7.0};
    xt::xarray<double> res = xt::view(arr1, 1) + arr2;
    std::cout << res << std::endl;
}

void ex2_run() {
    xt::xarray<int> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};
    arr.reshape({3, 3});
    std::cout << arr << std::endl;
}

void ex3_run() {
    xt::xarray<double> arr1{{1.0, 2.0, 3.0}, {2.0, 5.0, 7.0}, {2.0, 5.0, 7.0}};
    std::cout << arr1(0, 0) << std::endl;
    xt::xarray<int> arr2{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::cout << arr2(0) << std::endl;
}

void ex4_run() {
    xt::xarray<double> arr1{1.0, 2.0, 4.0};
    xt::xarray<unsigned int> arr2{2, 3, 4, 5, 6, 7};
    arr2.reshape({6, 1});
    xt::xarray<double> res = xt::pow(arr1, arr2);
    std::cout << arr1 << std::endl;
    std::cout << arr2 << std::endl;
    std::cout << res << std::endl;
}

void ex1_vec_run() {
    std::vector<double> v = {1., 2., 3., 4., 5., 6.};
    std::vector<std::size_t> shape = {2, 3};
    auto a1 = xt::adapt(v, shape);
    xt::xarray<double> a2 = {{1., 2., 3.}, {4., 5., 6.}};
    xt::xarray<double> res = a1 + a2;
    std::cout << res << std::endl;
    // a1(0, 0) = 20.;
}

void ex2_vec_run() {
    double v[6] = {1., 2., 3., 4., 5., 6.};
    std::vector<std::size_t> shape = {2, 3};
    auto a1 = xt::adapt(v, shape);
    xt::xarray<double> a2 = {{1., 2., 3.}, {4., 5., 6.}};
    xt::xarray<double> res = 2 * a1 + a2;
    std::cout << res << std::endl;
}

void ex3_vec_run() {
    xt::xarray<int> a1 = {1, 0, 1, 0, 0, 1, 0, 1};
    xt::xarray<int> a2 = {1, 1, 1, 1, 0, 0, 0, 0};
    xt::xarray<int> b1 = a1 & a2;
    xt::xarray<int> b2 = a1 | a2;
    xt::xarray<int> b3 = a1 ^ a2;
    xt::xarray<int> b4 = ~a1;
    xt::xarray<int> b5 = xt::left_shift(a1, a2);
    xt::xarray<int> b6 = xt::right_shift(a1, a2);
    std::cout << a1 << std::endl
              << a2 << std::endl
              << b1 << std::endl
              << b2 << std::endl
              << b3 << std::endl
              << b4 << std::endl
              << b5 << std::endl
              << b6 << std::endl;
    /*
        a1: {1, 0, 1, 0, 0, 1, 0, 1}
        a2: {1, 1, 1, 1, 0, 0, 0, 0}
        b1: {1, 0, 1, 0, 0, 0, 0, 0}
        b2: {1, 1, 1, 1, 0, 1, 0, 1}
        b3: {0, 1, 0, 1, 0, 1, 0, 1}
        b4: {-2, -1, -2, -1, -1, -2, -1, -2}
        b5: {2, 0, 2, 0, 0, 1, 0, 1}
        b6: {0, 0, 0, 0, 0, 1, 0, 1}
    */
}

void ex4_vec_run() {
    xt::xarray<bool> a1 = {1, 0, 1, 0, 0, 1, 0, 1};
    xt::xarray<bool> a2 = {1, 1, 1, 1, 0, 0, 0, 0};
    xt::xarray<bool> b1 = a1 & a2;
    xt::xarray<bool> b2 = a1 | a2;
    xt::xarray<bool> b3 = a1 ^ a2;
    xt::xarray<bool> b4 = ~a1;
    xt::xarray<bool> b5 = xt::left_shift(a1, a2);
    xt::xarray<bool> b6 = xt::right_shift(a1, a2);
    std::cout << a1 << std::endl
              << a2 << std::endl
              << b1 << std::endl
              << b2 << std::endl
              << b3 << std::endl
              << b4 << std::endl
              << b5 << std::endl
              << b6 << std::endl;
    /*
        { true, false,  true, false, false,  true, false,  true}
        { true,  true,  true,  true, false, false, false, false}
        { true, false,  true, false, false, false, false, false}
        { true,  true,  true,  true, false,  true, false,  true}
        {false,  true, false,  true, false,  true, false,  true}
        {false,  true, false,  true,  true, false,  true, false}
        { true, false,  true, false, false,  true, false,  true}
        {false, false, false, false, false,  true, false,  true}
    */
}

void ex1_view_run() {
    xt::xarray<int> a = xt::linspace<int>(0, 23, 24);
    a.reshape({3, 2, 4});
    std::cout << a << std::endl;
    // View with same number of dimensions
    auto v1 = xt::view(a, xt::range(1, 3), xt::all(), xt::range(1, 3));
    // => v1.shape() = { 2, 2, 2 }
    // => v1(0, 0, 0) = a(1, 0, 1)
    // => v1(1, 1, 1) = a(2, 1, 2)
    std::cout << v1 << std::endl;
    // View reducing the number of dimensions
    auto v2 = xt::view(a, 1, xt::all(), xt::range(0, 4, 2));
    // => v2.shape() = { 2, 2 }
    // => v2(0, 0) = a(1, 0, 0)
    // => v2(1, 1) = a(1, 1, 2)
    std::cout << v2 << std::endl;
    // View increasing the number of dimensions
    auto v3 = xt::view(a, xt::all(), xt::all(), xt::newaxis(), xt::all());
    // => v3.shape() = { 3, 2, 1, 4 }
    // => v3(0, 0, 0, 0) = a(0, 0, 0)
    std::cout << v3 << std::endl;
    // View with non contiguous slices
    auto v4 = xt::view(a, xt::drop(0), xt::all(), xt::keep(0, 3));
    // => v4.shape() = { 2, 2, 2 }
    // => v4(0, 0, 0) = a(1, 0, 0)
    // => v4(1, 1, 1) = a(2, 1, 3)
    std::cout << v4 << std::endl;
    // View built with negative index
    auto v5 = xt::view(a, -2, xt::all(), xt::range(0, 4, 2));
    // => v5 == v2
    std::cout << v5 << std::endl;
}

void ex1_ind_run() {
    xt::xarray<size_t> a = xt::arange<size_t>(3 * 4);
    a.reshape({3, 4});
    std::cout << a << std::endl;
    auto idx = xt::from_indices(xt::argwhere(a % 2));
    std::cout << idx << std::endl;
    auto idx_f = xt::ravel_indices(xt::argwhere(a % 2), a.shape());
    std::cout << idx_f << std::endl;
}

void ex2_ind_run() {
    xt::xarray<int> a = xt::arange<size_t>(24);
    std::cout << xt::print_options::line_width(160) << a << std::endl;
    // xt::xarray<int>
    auto sv = xt::strided_view(a, {xt::range(0, 24, 2)});
    sv *= 2;
    std::cout << xt::print_options::line_width(160) << sv << std::endl;
    std::cout << xt::print_options::line_width(160) << a << std::endl;
}

void ex3_ind_run() {
    xt::random::seed(0x123456);
    xt::xarray<std::complex<double>> a = xt::zeros<std::complex<double>>({24});
    xt::real(a) = xt::random::randn({24}, 0.0, 1.0);
    xt::imag(a) = xt::random::randn({24}, 0.0, 1.0);
    std::cout << xt::print_options::line_width(160) << a << std::endl;
    auto sv = xt::strided_view(a, {xt::range(0, 24, 2)});
    sv *= -2.0;
    std::cout << xt::print_options::line_width(160) << sv << std::endl;
    std::cout << xt::print_options::line_width(160) << a << std::endl;
}

void ex1_rnd_run() {
    // Note! Different from NumPy, the first argument is the shape of the output array!
    // Using a random number function from xtensor actually returns a lazy generator. That
    // means, accessing the same element of a random number generator does not give the
    // same random number if called twice!
    xt::random::seed(0x123456);
    // xt::random::seed(time(NULL));
    xt::xarray<double> a1 = xt::random::randn({4, 3}, 0.0, 1.0);
    std::cout << a1 << std::endl;
    xt::xarray<double> a2 = xt::random::randint({4, 3}, 0, 1 << 8);
    std::cout << a2 << std::endl;
    xt::random::shuffle(a2);
    std::cout << a2 << std::endl;
    auto shape = a2.shape();
    std::cout << xt::adapt(shape) << std::endl;
}

void ex1_csv_run() {
    std::cout << "Current path is: " << fs::current_path() << '\n';
    std::ifstream in_file;
    // in_file.open("/home/peva/projects/xt-expo/tv0/info_bits.txt");
    in_file.open("./../tv0/info_bits.txt");
    auto info_bits = xt::load_csv<short>(in_file);
    std::cout << xt::transpose(info_bits) << std::endl;
}

void ex1_qck_run() {
    xt::xarray<double> a0 = xt::arange<double>(0., 6.).reshape({2, 3});
    std::cout << a0 << std::endl;

    xt::xarray<double> a1 = {{1, 2, 3}};
    xt::xarray<double> b2 = {{2, 3, 4}};
    auto c0 = xt::concatenate(xt::xtuple(a1, b2));
    std::cout << c0 << std::endl;
    auto c1 = xt::concatenate(xt::xtuple(a1, b2), 1);
    std::cout << c1 << std::endl;
}

void ex1_red_run() {
    // Sum
    xt::xarray<int> a1 = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::sum(a1, {1});
    std::cout << r0 << std::endl;
    // Outputs {6, 15}
    xt::xarray<int> r1 = xt::sum(a1);
    std::cout << r1 << std::endl;
    // Outputs {21}, i.e. r1 is a 0D-tensor
    int r2 = xt::sum(a1)();
    std::cout << r2 << std::endl;
    // Outputs 21
    // Prod
    xt::xarray<int> a2 = {{1, 2}, {3, 4}};
    xt::xarray<int> r3 = xt::prod(a2, {1});
    std::cout << r3 << std::endl;
    xt::xarray<int> r4 = xt::prod(a2);
    std::cout << r4 << std::endl;
    int r5 = xt::prod(a2)();
    std::cout << r5 << std::endl;
    xt::xarray<double> b0 = xt::cumsum(a1, 1);
    std::cout << b0 << std::endl;
}

void ex1_man_run() {
    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto t0 = xt::roll(a, 2);
    std::cout << t0 << std::endl;
    auto t1 = xt::roll(a, 2, 1);
    std::cout << t1 << std::endl;
}

void ex1_crc_run() {
    std::cout << "Current path is: " << fs::current_path() << '\n';
    std::ifstream in_file;
    in_file.open("./../dl/tv0/info_bits.txt");
    xt::xarray<int> msg = xt::ravel(xt::load_csv<short>(in_file));
    std::cout << "msg:" << std::endl << xt::transpose(msg) << std::endl;

    // clang-format off
    xt::xarray<int> crc_r_v = {
        0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    // clang-format on

    int step = 32;
    int pad = step - msg.size();

    xt::xarray<int> crc_r = crc_r_v.reshape({step, step});
    std::cout << xt::print_options::line_width(160) << xt::print_options::edge_items(20)
              << "crc_r:" << std::endl
              << crc_r << std::endl;

    xt::xarray<int> crc_s = xt::zeros<int>({step});
    std::cout << xt::print_options::line_width(160) << "crc_s:" << std::endl
              << xt::transpose(crc_s) << std::endl;

    xt::xarray<int> p_msg = xt::concatenate(xt::xtuple(msg, xt::zeros<int>({pad})));
    std::cout << xt::print_options::line_width(160) << "p_msg:" << std::endl
              << xt::transpose(p_msg) << std::endl;

    xt::xarray<int> crc_p = crc_s ^ p_msg;
    std::cout << xt::print_options::line_width(160) << "crc_p:" << std::endl
              << xt::transpose(crc_p) << std::endl;

    for (int j = 0; j < step; ++j) {
        auto crc_r_r = xt::view(crc_r, j, xt::all());
        // std::cout << xt::print_options::line_width(160) << "crc_r_r:" << std::endl
        //           << xt::transpose(crc_r_r) << std::endl;
        xt::xarray<int> crc_v = crc_r_r & crc_p;
        // std::cout << xt::print_options::line_width(160) << "crc_v:" << std::endl
        //           << xt::transpose(crc_v) << std::endl;
        crc_s(j) = xt::sum<int>(crc_v)(0);
    }

    crc_s %= 2;
    std::cout << xt::print_options::line_width(160) << "crc_s:" << std::endl
              << xt::transpose(crc_s) << std::endl;
}

void ex2_crc_run() {
    std::cout << "Current path is: " << fs::current_path() << '\n';

    // Read message
    std::ifstream in_file;
    in_file.open("./../dl/tv0/info_bits.txt");
    xt::xarray<int> msg = xt::ravel(xt::load_csv<short>(in_file));
    std::cout << "msg:" << std::endl << xt::transpose(msg) << std::endl;

    // CRC vector for remainder
    // clang-format off
    xt::xarray<int> crc_r_v = {
        0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    // clang-format on

    // CRC matrix
    int step = 32;
    int pad = step - msg.size();
    xt::xarray<int> crc_r = crc_r_v.reshape({step, step});
    std::cout << xt::print_options::line_width(160) << xt::print_options::edge_items(20)
              << "crc_r:" << std::endl
              << crc_r << std::endl;

    // CRC state
    xt::xarray<int> crc_s = xt::zeros<int>({step});
    std::cout << xt::print_options::line_width(160) << "crc_s:" << std::endl
              << xt::transpose(crc_s) << std::endl;

    // Input padding to step size
    xt::xarray<int> p_msg = xt::concatenate(xt::xtuple(msg, xt::zeros<int>({pad})));
    std::cout << xt::print_options::line_width(160) << "p_msg:" << std::endl
              << xt::transpose(p_msg) << std::endl;

    // XOR stage
    xt::xarray<int> crc_p = crc_s ^ p_msg;
    std::cout << xt::print_options::line_width(160) << "crc_p:" << std::endl
              << xt::transpose(crc_p) << std::endl;

    // CRC computation with matrix times vector
    crc_s = xt::linalg::dot(crc_r, crc_p);
    crc_s %= 2;
    std::cout << xt::print_options::line_width(160) << "crc_s:" << std::endl
              << xt::transpose(crc_s) << std::endl;
}

void ex1_mpow_run() {
    xt::xarray<double> arr1{{1, 1, 0}, {1, 0, 1}, {0, 0, 0}};
    std::cout << "arr1:" << std::endl << arr1 << std::endl;
    for (long n = 2; n < 8; ++n) {
        xt::xarray<double> arr_n = xt::linalg::matrix_power(arr1, n);
        std::cout << "arr^" << n << ":" << std::endl << arr_n << std::endl;
    }
}

void ex1_cmplx_run() {
    // Note! Different from NumPy, the first argument is the shape of the output array!
    // Using a random number function from xtensor actually returns a lazy generator. That
    // means, accessing the same element of a random number generator does not give the
    // same random number if called twice!
    xt::random::seed(0x123456);
    // xt::random::seed(time(NULL));
    xt::xarray<std::complex<double>> a1 = xt::zeros<std::complex<double>>({4, 3});
    xt::real(a1) = xt::random::randn({4, 3}, 0.0, 1.0);
    xt::imag(a1) = xt::random::randn({4, 3}, 0.0, 1.0);
    std::cout << a1 << std::endl;
}

void ex2_cmplx_run() {
    xt::random::seed(0x123456);
    xt::xarray<std::complex<double>> a1 = xt::zeros<std::complex<double>>({8});
    xt::real(a1) = xt::random::randn({8}, 0.0, 1.0);
    xt::imag(a1) = xt::random::randn({8}, 0.0, 1.0);
    std::cout << xt::print_options::line_width(160) << "a1:" << std::endl
              << a1 << std::endl;
    auto a1_fft = xt::fftw::fft(a1);
    std::cout << xt::print_options::line_width(160) << "a1_fft:" << std::endl
              << a1_fft << std::endl;
    auto a1_fftshift = xt::fftw::fftshift(a1_fft);
    std::cout << xt::print_options::line_width(160) << "a1_fftshift:" << std::endl
              << a1_fftshift << std::endl;
}