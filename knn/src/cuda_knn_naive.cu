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
__host__ void CUDAKNNNaive<T>::find(int query, int k, std::vector<int>& knn)
{
    int N_threads = this->_bytes_size / (this->_dim * sizeof(T));
    int blockPerGrid, threadsPerBlock;

    query = query * this->_dim;

    get_dim_comp_dist<T>(N_threads, blockPerGrid, threadsPerBlock);
    comp_dist<T>(blockPerGrid, threadsPerBlock, this->_data, this->_dim, query, 
            this->_dev_sort, this->_bytes_size);
        
    knn.resize(k);
    CUDA_ERR(cudaMemcpy(knn.data(), this->_dev_sort._value, k*sizeof(int), 
                cudaMemcpyDeviceToHost));
}

////////////////////////////////////////////////////////////////////////////////////////

