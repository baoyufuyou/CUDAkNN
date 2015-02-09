/*
 * =====================================================================================
 *       Filename:  cuda_knn_thrust.cpp
 *    Description:  
 *        Created:  2015-02-09 10:30
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn.hpp"

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
void comp_dist(uint blockPerGrid, uint threadsPerBlock, T* dev_data, uint dim, 
        uint query, struct sort_t<T> dev_sort)
{

    __comp_dist<T><<<blockPerGrid, threadsPerBlock>>>(dev_data, dim, query, dev_sort);
    CUDA_ERR(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void get_dim_comp_dist(int N_threads, int &blocksPerGrid, int &threadsPerBlock)
{
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, 
            (void*)__comp_dist<T>, 0, N_threads);

    blocksPerGrid = (N_threads + threadsPerBlock - 1) / threadsPerBlock;
}

////////////////////////////////////////////////////////////////////////////////////////

template void comp_dist<float>(uint, uint, float*, uint, uint, struct sort_t<float>);
template void get_dim_comp_dist<float>(int, int&, int&);

////////////////////////////////////////////////////////////////////////////////////////


