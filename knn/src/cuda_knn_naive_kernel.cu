/*
 * =====================================================================================
 *       Filename:  cuda_knn_naive.cu
 *    Description:  
 *        Created:  2015-02-03 19:18
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn_naive.hpp"

////////////////////////////////////////////////////////////////////////////////////////
/**
 * @arg0: raw data in the device
 * @arg1: space dimentions
 * @arg2: DIRECT index of the query in the raw data
 * @arg3: return array containing distances from each data to query 
 * */
template <typename T> 
__global__ void __comp_dist(T* dev_data, uint dim, uint query, struct sort_t<T>* dev_sort)
{
    uint k = blockDim.x * blockIdx.x + threadIdx.x;
    T res_local = 0;
    T res_query_local = 0;
    T res_index_local = 0;

    uint index = k * dim;

    dev_sort[k].index = k;
    
    for(uint i=0; i < dim; i++)
    {
        res_query_local = dev_data[query + i];
        res_index_local = dev_data[index + i];
        res_local += (res_query_local - res_index_local) * (res_query_local - res_index_local);
    }

    dev_sort[k].key = res_local;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ void comp_dist(uint blocksPerGrid, uint threadsPerBlock, T* dev_data, uint dim, 
        uint query, struct sort_t<T>* dev_sort)
{
    __comp_dist<T><<<blocksPerGrid, threadsPerBlock>>>(dev_data, dim, query, dev_sort);
    CUDA_ERR(cudaGetLastError());
}


// Overload
template <typename T>
void comp_dist(uint N_threads, T* dev_data, uint dim, uint query, 
        struct sort_t<T>* dev_sort)
{
    int minGridSize, gridSize;
    int blockSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)__comp_dist<T>, 0, N_threads);

    gridSize = (N_threads + blockSize - 1) / blockSize;

    comp_dist<T>(gridSize, blockSize, dev_data, dim, query, dev_sort);
}


// explicit to compile template for float
template void comp_dist<float>(uint blocksPerGrid, uint threadsPerBlock, float* dev_data, 
        uint dim, uint query, struct sort_t<float>* dev_sort);

template void comp_dist<float>(uint N_threads, float* dev_data, uint dim, uint query, 
        struct sort_t<float>* dev_sort);

////////////////////////////////////////////////////////////////////////////////////////
