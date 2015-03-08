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
void CUDAKNNThrust<T>::find(int query, int k, std::vector<int>& knn)
{
    int N_threads = this->_bytes_size / (this->_dim * sizeof(T));
    int blockPerGrid, threadsPerBlock;

    query = query * this->_dim;

    get_dim_comp_dist<T>(N_threads, blockPerGrid, threadsPerBlock);
    comp_dist<T>(blockPerGrid, threadsPerBlock, this->_data, this->_dim, query, 
            this->_dev_sort, this->_bytes_size);
    //comp_dist_opt<T>(this->_data, this->_dim, query, this->_dev_sort, this->_bytes_size);
    
    thrust::device_ptr<T> key(this->_dev_sort._key);
    thrust::device_ptr<int> value(this->_dev_sort._value);
    thrust::sort_by_key(key, key + N_threads, value); // OPT !
    
    knn.resize(k);
    CUDA_ERR(cudaMemcpy(knn.data(), this->_dev_sort._value + 1, k*sizeof(int), 
                cudaMemcpyDeviceToHost));

    return;
}

////////////////////////////////////////////////////////////////////////////////////////
