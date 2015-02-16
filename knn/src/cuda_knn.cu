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
__global__ void __comp_dist(T* dev_data, int dim, int query, struct sort_t<T> dev_sort, 
        int dev_data_bytes_size)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    T res_local = 0;
    T res_query_local = 0;
    T res_index_local = 0;
    
    int index = k * dim;

    if(index < dev_data_bytes_size / sizeof(T))
    {
        dev_sort._value[k] = k;

        for(int i=0; i < dim; i++)
        {
            res_query_local = dev_data[query + i];
            res_index_local = dev_data[index + i];
            res_local += (res_query_local - res_index_local) * (res_query_local - res_index_local);
        }

        dev_sort._key[k] = res_local;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Wrapper
template <typename T>
void comp_dist(int blockPerGrid, int threadsPerBlock, T* dev_data, int dim, 
        int query, struct sort_t<T> dev_sort, int dev_data_bytes_size)
{
    __comp_dist<T><<<blockPerGrid, threadsPerBlock>>>(dev_data, dim, query, dev_sort, 
            dev_data_bytes_size);
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

template void comp_dist<float>(int, int, float*, int, int, struct sort_t<float>, int);
template void get_dim_comp_dist<float>(int, int&, int&);

////////////////////////////////////////////////////////////////////////////////////////
/*
 * @arg0: raw data in the device
 * @arg1: space dimentions
 * @arg2: DIRECT index of the query in the raw data
 * @arg3: return array containing distances from each data to query 
 * */
template <typename T>
__global__ void __comp_dist_opt(T* dev_data, int dim, int query, struct sort_t<T> dev_sort,
        int dev_data_bytes_size)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ T res_shared;
    T res_query_local = 0;
    T res_index_local = 0;

    res_shared = 0;

    __syncthreads(); // res_shared must be equaly zero to all threads
        
    int index = k;

    res_query_local = dev_data[query + threadIdx.x];
    res_index_local = dev_data[k];

    atomicAdd(&res_shared, (res_query_local - res_index_local) * (res_query_local - res_index_local));

    if(threadIdx.x == 0) {
        dev_sort._value[index] = index;
        dev_sort._key[index] = res_shared;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ void comp_dist_opt(int blockPerGrid, int threadsPerBlock, T* dev_data, int dim, 
        int query, struct sort_t<T> dev_sort, int dev_data_bytes_size)
{
    __comp_dist_opt<T><<<blockPerGrid, threadsPerBlock>>>(dev_data, dim, query, dev_sort, 
            dev_data_bytes_size);
    CUDA_ERR(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ void get_dim_comp_dist_opt(int N_threads, int &blocksPerGrid, int &threadsPerBlock)
{
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, 
            (void*)__comp_dist_opt<T>, 0, N_threads);

    blocksPerGrid = (N_threads + threadsPerBlock - 1) / threadsPerBlock;
}

////////////////////////////////////////////////////////////////////////////////////////

template void comp_dist_opt<float>(int, int, float*, int, int, struct sort_t<float>, int);
template void get_dim_comp_dist_opt<float>(int, int&, int&);

////////////////////////////////////////////////////////////////////////////////////////

