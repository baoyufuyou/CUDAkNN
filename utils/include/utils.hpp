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

#include <chrono>
#include <string>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////
// Types
typedef unsigned char uchar;
typedef unsigned int  uint;
typedef unsigned short int usint;

////////////////////////////////////////////////////////////////////////////////////////
// Global variables for time_point
static std::chrono::high_resolution_clock::time_point t0;
static std::chrono::high_resolution_clock::time_point t1;

////////////////////////////////////////////////////////////////////////////////////////
// Macros for time
#define TIME_BETWEEN(code) \
    t0 = std::chrono::high_resolution_clock::now(); \
    code \
    t1 = std::chrono::high_resolution_clock::now(); \
    std::cout << "Code took " << (std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0)).count(); \
    std::cout << "s to run: " << std::endl; \
    std::cout << "---------------------------------------------------------" << std::endl; \
    std::cout << #code << std::endl; \
    std::cout << "---------------------------------------------------------" << std::endl;

////////////////////////////////////////////////////////////////////////////////////////

#define EPSILON 1e-5

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !UTILS_HPP */

////////////////////////////////////////////////////////////////////////////////////////

