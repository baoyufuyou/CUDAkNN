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
void CUDAKNNNaive<T>::find(uint query, uint k, std::vector<uint>& knn)
{
    uint N_threads = this->_data.size() / this->_dim;
    uint threadsPerBlock = 256;
    uint blocksPerGrid = (N_threads + threadsPerBlock - 1) / threadsPerBlock;

    query = query * this->_dim;

 //   comp_dist<float>(blocksPerGrid, threadsPerBlock, this->_dev_data, 
 //           this->_dim, query, this->_dev_sort);
    comp_dist<T>(N_threads, this->_dev_data, this->_dim, query, this->_dev_sort);

}

////////////////////////////////////////////////////////////////////////////////////////


