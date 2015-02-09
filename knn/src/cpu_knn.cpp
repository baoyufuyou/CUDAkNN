/*
 * =====================================================================================
 *       Filename:  cpu_knn.cpp
 *    Description:  
 *        Created:  2015-02-02 12:16
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cpu_knn.hpp"

#include <algorithm>

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Implementation of the method of the supper class KNN  
 * */
template <typename T>
void CPUKNN<T>::find(uint query, uint k, std::vector<uint>& knn)
{
    uint dist_size = this->_bytes_size / (this->_dim * sizeof(T)); // number different elements in the _data vector
    std::vector<std::pair<T, uint>> dist(dist_size); // distance vector

    knn.clear(); knn.resize(k); // vector will have k indices of the k-nearest neighbors

    /* Computes distances */
    for(uint i=0; i < dist_size; i++)
    {
        dist[i].first = norm2_squared_dist(query, i);
        dist[i].second = i;
    }
    
    /* Sorts the distances */
    std::sort(dist.begin(), dist.end(), &CPUKNN::cmp_pair);

    /* Gets the index of k-nearest neighbors */
    for(uint i=1; i <= k && i < dist_size; i++)
    {
        knn[i-1] = dist[i].second;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Evaluates the squared distance using the norm 2  
 * */
template <typename T>
T CPUKNN<T>::norm2_squared_dist(uint a, uint b)
{
    T norm = 0;
    T vec;

    a = this->_dim * a;
    b = this->_dim * b;
    
    for(uint i=0; i < this->_dim; i++) 
    {
        vec = (this->_data[a + i] - this->_data[b + i]);
        norm += vec * vec;
    }

    return norm;
}

////////////////////////////////////////////////////////////////////////////////////////
