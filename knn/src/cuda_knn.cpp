/*
 * =====================================================================================
 *       Filename:  cuda_knn.cpp
 *    Description:  
 *        Created:  2015-02-02 11:28
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn.hpp"

#include <cuda_runtime.h>

#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUDAKNN<T>::CUDAKNN(uint dim, std::vector<T>& data) 
    : 
        KNN<T>(dim, data),
        _dev_data(NULL)
{
    T* host_data = this->_data.data();

    _data_size_byte = this->_data.size() * sizeof(T);

    CUDA_ERR(cudaMalloc((void**)&_dev_data, _data_size_byte));
    CUDA_ERR(cudaMemcpy(_dev_data, host_data, _data_size_byte, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUDAKNN<T>::CUDAKNN(uint dim, uint data_size_byte, T* dev_data)
    :
        KNN<T>(dim, *(new std::vector<T>(0))),
        _dev_data(dev_data),
        _data_size_byte(data_size_byte)
{
    /* Nothing to do here */
}

////////////////////////////////////////////////////////////////////////////////////////
