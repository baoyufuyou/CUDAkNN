/*
 * =====================================================================================
 *       Filename:  error.hpp
 *    Description:  
 *        Created:  2015-02-02 11:13
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef ERROR_HPP
#define ERROR_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include <string>

////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * Prints error message and exits  
 * */
 void __error(const std::string& error_code, const std::string& file, int line);

/**
 * Prints warning message and continues execution
 * */
 void __warning(const std::string& error_code, const std::string& file, int line);

#ifdef __cplusplus
};
#endif

////////////////////////////////////////////////////////////////////////////////////////

#ifdef NDEBUG
#define FATAL_ERROR(str)
#define WARNING_ERROR(str)
#define ASSERT_FATAL_ERROR(boolean, str)
#define ASSERT_WARNING_ERROR(boolean, str)
#else
#define FATAL_ERROR(str) __error(str, __FILE__, __LINE__)
#define WARNING_ERROR(str) __warning(str, __FILE__, __LINE__)
#define ASSERT_FATAL_ERROR(boolean, str) (void)((boolean) || (__error(str, __FILE__, __LINE__),0))
#define ASSERT_WARNING_ERROR(boolean, str) (void)((boolean) || (__warning(str, __FILE__, __LINE__),0))
#endif

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !ERROR_HPP */

////////////////////////////////////////////////////////////////////////////////////////

