/*
 * =====================================================================================
 *       Filename:  cuda_knn_thrust.cpp
 *    Description:  
 *        Created:  2015-02-09 10:30
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn_thrust.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CUDAKNNThrust<T>::find(uint query, uint k, std::vector<uint>& knn)
{
    int blockPerGrid, threadsPerBlock;
    int N_threads = this->_bytes_size / (this->_dim * sizeof(T));

    query = query * this->_dim;

    get_dim_comp_dist<T>(N_threads, blockPerGrid, threadsPerBlock);
    comp_dist<T>(blockPerGrid, threadsPerBlock, this->_data, this->_dim, query, 
            this->_dev_sort);
    
    thrust::device_ptr<T> key(this->_dev_sort._key);
    thrust::device_ptr<uint> value(this->_dev_sort._value);
    thrust::sort_by_key(key, key + N_threads, value);
    
    knn.resize(k);
    CUDA_ERR(cudaMemcpy(knn.data(), this->_dev_sort._value + 1, k*sizeof(uint), 
                cudaMemcpyDeviceToHost));

    return;
}

////////////////////////////////////////////////////////////////////////////////////////
