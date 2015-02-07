/*
 * =====================================================================================
 *       Filename:  parser.cpp
 *    Description:  
 *        Created:  2015-02-07 15:08
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "parser.hpp"

#include <iostream>
#include <cstring>
#include <string>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser(int argc, char* argv[])
{
    for(int i=1; i < argc; i++)
    {
        _argv << argv[i] << " ";
    }
}

////////////////////////////////////////////////////////////////////////////////////////

inline void Parser::print_help() const
{
    std::cout << HELP_STR << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////

void Parser::get_next_option(enum options_t& op, uint& q)
{
    char option[MAX_INPUT_SIZE];

    _argv.getline(option, MAX_INPUT_SIZE, '-');
    _argv.getline(option, MAX_INPUT_SIZE, ' ');
    
    if(strcmp(option, "n") == 0) {
        _argv.getline(option, MAX_INPUT_SIZE, ' ');
        q = std::stoul(option);
        op = SET_N_POINTS;
    }
    else if(strcmp(option, "d") == 0) {
        _argv.getline(option, MAX_INPUT_SIZE, ' ');
        q = std::stoul(option);
        op = SET_POINTS_DIM;
    }
    else if(strcmp(option, "k") == 0) {
        _argv.getline(option, MAX_INPUT_SIZE, ' ');
        q = std::stoul(option);
        op = SET_K;
    }
    else if(strcmp(option, "q") == 0) {
        _argv.getline(option, MAX_INPUT_SIZE, ' ');
        q = std::stoul(option);
        op = SET_QUERY;
    }
    else if(strcmp(option, "h") == 0) {
        op = PRINT_HELP;
    }
    else if(strcmp(option, "") == 0) {
        op = EOF_STRING;   
    }
    else {
        op = UNDEFINED_OPTION;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

void Parser::get_options(uint& n_points, uint& dim, uint& k, uint& query)
{
    enum options_t opt;
    uint opt_val;

    // Default values
    n_points = 10;
    dim = 1;
    k = 1;
    query = 0;

    // Read input
    while(!_argv.eof())
    {
        get_next_option(opt, opt_val);

        switch(opt)
        {
            case SET_N_POINTS:
                n_points = opt_val;
                break;
            case SET_POINTS_DIM:
                dim = opt_val;
                break;
            case SET_K:
                k = opt_val;
                break;
            case SET_QUERY:
                query = opt_val;
                break;
            case PRINT_HELP:
                print_help();
                exit(EXIT_SUCCESS);
                break;
            case EOF_STRING:
                return;
            case UNDEFINED_OPTION:
            default:
                print_help();
                exit(EXIT_FAILURE);
        }
    }
}



////////////////////////////////////////////////////////////////////////////////////////

