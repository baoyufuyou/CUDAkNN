/*
 * =====================================================================================
 *       Filename:  cuda_knn_naive.cpp
 *    Description:  
 *        Created:  2015-02-03 19:18
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn_naive.hpp"

#include <cuda_runtime.h>

#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUDAKNNNaive<T>::CUDAKNNNaive(uint dim, std::vector<T>& data) 
    : 
        CUDAKNN<T>(dim, data)
{
    _dev_sort_byte = (this->_data.size() / this->_dim) * sizeof(struct sort_t<T>);

    CUDA_ERR(cudaMalloc((void**)&_dev_sort, _dev_sort_byte));
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUDAKNNNaive<T>::CUDAKNNNaive(uint dim, uint data_size_byte, T* _dev_data)
    :
        CUDAKNN<T>(dim, data_size_byte, _dev_data)
{
    _dev_sort_byte = (this->_data.size() / this->_dim) * sizeof(struct sort_t<T>);

    CUDA_ERR(cudaMalloc((void**)&_dev_sort, _dev_sort_byte));
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CUDAKNNNaive<T>::find(uint query, uint k, std::vector<uint>& knn)
{
    uint N_threads = this->_data.size() / this->_dim;
    uint threadsPerBlock = 256;
    uint blocksPerGrid = (N_threads + threadsPerBlock - 1) / threadsPerBlock;

    query = query * this->_dim;

 //   comp_dist<float>(blocksPerGrid, threadsPerBlock, this->_dev_data, 
 //           this->_dim, query, this->_dev_sort);
    comp_dist<float>(N_threads, this->_dev_data, this->_dim, query, this->_dev_sort);
}

////////////////////////////////////////////////////////////////////////////////////////
