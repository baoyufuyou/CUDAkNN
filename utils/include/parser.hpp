/*
 * =====================================================================================
 *       Filename:  parser.hpp
 *    Description:  
 *        Created:  2015-02-07 15:05
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef PARSER_HPP
#define PARSER_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////////////

#define MAX_INPUT_SIZE 256

////////////////////////////////////////////////////////////////////////////////////////

#define HELP_STR \
"Usage: cudaknn [options] \n\
Options: \n\
  -n: Number of random points to generate \n\
  -d: Number of dimentions of each point \n\
  -k: Number of nearest neighbors to find \n\
  -q: Index of the query point (must be smaller than n_points) \n\
  -p: Print information about CUDA enviroment \n\
  -h: Display this information \n\
\n\
CUDA KNN - A simple implementation of k-nearest neighbors in CUDA\n\
Authors: \n\
    Tiago Lobato Gimenes - (tlgimenes@gmail.com)\n\
    Carlos Coelho Lechner - (cclechner@gmail.com)" 

////////////////////////////////////////////////////////////////////////////////////////

enum options_t
{
    SET_N_POINTS,
    SET_POINTS_DIM,
    SET_K,
    SET_QUERY,
    PRINT_CUDA_ENVIROMENT,
    PRINT_HELP,
    EOF_STRING,
    UNDEFINED_OPTION
};

////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
    private:
        std::stringstream _argv; // main's argv

    protected:
        void get_next_option(enum options_t&, int&);

        inline void print_help() const;

    public:
        Parser(int argc, char* argv[]);

        /**
         * Get the program options from the argv. If you want to add any new
         * parameter for the program, add this parameter here !
         * */
        void get_options(int& n_points, 
                         int& dim,
                         int& k,
                         int& query);

        /* 
         * Gets 
         * */
        inline const std::stringstream& argv() const {return _argv;}

        /**
         * Sets 
         *  */
        inline std::stringstream& argv() {return _argv;}
};

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !PARSER_HPP */

////////////////////////////////////////////////////////////////////////////////////////

