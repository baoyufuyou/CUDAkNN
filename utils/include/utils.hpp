/*
 * =====================================================================================
 *       Filename:  utils.hpp
 *    Description:  
 *        Created:  2015-02-02 11:54
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef UTILS_HPP
#define UTILS_HPP

////////////////////////////////////////////////////////////////////////////////////////

#define CPP99 199711L

////////////////////////////////////////////////////////////////////////////////////////

#if __cplusplus > CPP99
#include <chrono>
#endif
#include <string>
#include <iostream>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////
// Types
typedef unsigned char uchar;
typedef unsigned int  uint;
typedef unsigned short int usint;

////////////////////////////////////////////////////////////////////////////////////////
// Global variables for time_point
#if __cplusplus > CPP99
static std::chrono::high_resolution_clock::time_point t0;
static std::chrono::high_resolution_clock::time_point t1;
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Macros for time
#if __cplusplus > CPP99
#define TIME_BETWEEN(code) \
    t0 = std::chrono::high_resolution_clock::now(); \
    code \
    t1 = std::chrono::high_resolution_clock::now(); \
    std::cout << "Code took " << (std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0)).count(); \
    std::cout << "s to run: " << std::endl; \
    std::cout << "---------------------------------------------------------" << std::endl; \
    std::cout << #code << std::endl; \
    std::cout << "---------------------------------------------------------" << std::endl;
#else
#include "error.hpp"
#define TIME_BETWEEN(code) WARNING_ERROR("TIME_BETWEEN needs at least C++11 compliant compiler")
#endif


////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
extern std::vector<T>& gen_random_data(uint n_points, uint dimention);

////////////////////////////////////////////////////////////////////////////////////////

#define EPSILON 1e-5

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !UTILS_HPP */

////////////////////////////////////////////////////////////////////////////////////////

