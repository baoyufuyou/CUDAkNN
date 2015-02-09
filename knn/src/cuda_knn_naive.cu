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

/*
 * @arg0: raw data in the device
 * @arg1: space dimentions
 * @arg2: DIRECT index of the query in the raw data
 * @arg3: return array containing distances from each data to query 
 * */
template <typename T>
__global__ void __comp_dist(T* dev_data, uint dim, uint query, struct sort_t<T> dev_sort)
{
    uint k = blockDim.x * blockIdx.x + threadIdx.x;
    T res_local = 0;
    T res_query_local = 0;
    T res_index_local = 0;

    uint index = k * dim;

    dev_sort._value[k] = k;
    
    for(uint i=0; i < dim; i++)
    {
        res_query_local = dev_data[query + i];
        res_index_local = dev_data[index + i];
        res_local += (res_query_local - res_index_local) * (res_query_local - res_index_local);
    }

    dev_sort._key[k] = res_local;
}


////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ void CUDAKNNNaive<T>::find(uint query, uint k, std::vector<uint>& knn)
{
    int N_threads = this->_data.size() / this->_dim;
    int threadsPerBlock, minGridSize;
    int blocksPerGrid;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, 
            (void*)__comp_dist<T>, 0, N_threads);

    blocksPerGrid = (N_threads + threadsPerBlock - 1) / threadsPerBlock;

    query = query * this->_dim;

    __comp_dist<T><<<blocksPerGrid, threadsPerBlock>>>(this->_dev_data, this->_dim, query, 
            this->_dev_sort);
    CUDA_ERR(cudaGetLastError());

    knn.resize(k);
    CUDA_ERR(cudaMemcpy(knn.data(), this->_dev_sort._value, k*sizeof(uint), cudaMemcpyDeviceToHost));
}

////////////////////////////////////////////////////////////////////////////////////////

