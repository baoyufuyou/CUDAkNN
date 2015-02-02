/*
 * =====================================================================================
 *       Filename:  utils.cpp
 *    Description:  
 *        Created:  2015-02-02 17:19
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "utils.hpp"

#include <ctime>
#include <cstdlib>

#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::vector<T>& gen_random_data(uint n_points, uint dim)
{
    FATAL_ERROR("Function not implemented for the type specified");
}

////////////////////////////////////////////////////////////////////////////////////////

template <>
std::vector<float>& gen_random_data(uint n_points, uint dim)
{
    ASSERT_FATAL_ERROR(dim > 0 && n_points > 0, "Dimention must be greater than zero");

    std::vector<float>& data = *(new std::vector<float>(n_points * dim));

    srand(time(NULL));

    for(uint i=0; i < data.size(); i+=dim)
    {
        for(uint j=0; j < dim; j++)
        {
            data[i+j] = -1.0 + 2 * static_cast <float> (rand()) / 
                static_cast <float> (RAND_MAX); 
        }
    }

    return data;
}

////////////////////////////////////////////////////////////////////////////////////////

template <>
std::vector<int>& gen_random_data<int>(uint n_points, uint dim)
{
    ASSERT_FATAL_ERROR(dim > 0, "Dimention must be greater than zero");

    std::vector<int>& data = *(new std::vector<int>(n_points * dim));

    srand(time(NULL));

    for(uint i=0; i < data.size(); i+=dim)
    {
        for(uint j=0; j < dim; j++)
        {
            data[i+j] = rand(); 
        }
    }

    return data;
}

////////////////////////////////////////////////////////////////////////////////////////
