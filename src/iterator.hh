#include <exception>
#include <iostream>
#include <tuple>
#include <vector>

#include "timer.hh"

using std::cout;
using std::endl;

template <typename T>
std::vector<std::tuple<T>> Cartesian_product(T num) {
    std::vector<std::tuple<T>> vec;
    for (size_t i = 0; i < num; i++) vec.push_back(std::make_tuple(i));
    return vec;
}

template <typename T, typename... Args>
std::vector<std::tuple<T, Args...>> Cartesian_product(T num, Args... args) {
    std::vector<std::tuple<T, Args...>> vec;
    for (auto tuple : Cartesian_product(args...))
        for (size_t i = 0; i < num; i++) vec.push_back(std::tuple_cat(std::make_tuple(i), tuple));
    return vec;  // implicitly std::move
}

int physical() {
    return 0;
}

template <typename T>
void physical(T x) {
    throw std::runtime_error("Function arguments must be in even number!");
}

template <typename T, typename... Args>
T physical(T dim, T idx, Args... args) {
    return idx + dim * physical(args...);
}

int test_physical() {
    try {
        int d0 = 20, d1 = 10, d2 = 50;
        int i0 = 2, i1 = 4, i2 = 5;
        std::cout << physical(int{1}, i0, d0, i1, d1, i2) << std::endl;
    } catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
