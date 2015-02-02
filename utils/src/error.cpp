/*
 * =====================================================================================
 *       Filename:  error.cpp
 *    Description:  
 *        Created:  2015-02-02 11:15
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "error.hpp"

#include <iostream>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <ctime>

////////////////////////////////////////////////////////////////////////////////////////

void __error(const std::string& error_code, const std::string& file, int line)
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    struct tm *tm = std::localtime(&now_c);

    std::cout << "FATAL ERROR ! " << file << ":" << line; 
    std::cout << " ["<< tm->tm_mday << "/" << (tm->tm_mon+1) << "/" << (tm->tm_year+1900);
    std::cout << "]:" << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec << std::endl;
    std::cout << "'" << error_code << "'" << std::endl << std::endl;

    exit(EXIT_FAILURE);

    return;
}

////////////////////////////////////////////////////////////////////////////////////////

void __warning(const std::string& error_code, const std::string& file, int line)
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    struct tm *tm = std::localtime(&now_c);

    std::cout << "WARNING ! " << file << ":" << line; 
    std::cout << " ["<< tm->tm_mday << "/" << (tm->tm_mon+1) << "/" << (tm->tm_year+1900);
    std::cout << "]:" << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec << std::endl;
    std::cout << "'" << error_code << "'" << std::endl << std::endl;
    
    return;
}

////////////////////////////////////////////////////////////////////////////////////////
