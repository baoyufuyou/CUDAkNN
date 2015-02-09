/*
 * =====================================================================================
 *       Filename:  knn.cpp
 *    Description:  
 *        Created:  2015-02-02 12:13
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "knn.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void KNN<T>::reset(uint dim, T* data, size_t bytes_size)
{
    _dim = dim;
    _data = data;
    _bytes_size = bytes_size;
    ASSERT_FATAL_ERROR((bytes_size % _dim) == 0, \
            "data size must devide number of dimentions");
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void KNN<T>::find(uint query, uint k, std::vector<uint>& knn)
{
    FATAL_ERROR("Cannot find from raw knn");
}

////////////////////////////////////////////////////////////////////////////////////////

