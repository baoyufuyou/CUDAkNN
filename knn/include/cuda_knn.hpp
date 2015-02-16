/*
 * =====================================================================================
 *       Filename:  cuda_knn.hpp
 *    Description:  
 *        Created:  2015-02-02 11:28
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KNN_HPP
#define CUDA_KNN_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include "knn.hpp"
#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

enum mem_scope_t
{
    PTR_TO_HOST_MEM,
    PTR_TO_DEVICE_MEM
};

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct sort_t 
{
    inline sort_t(int nKeys) :
        _key(NULL), _value(NULL)
    {
        CUDA_ERR(cudaMalloc((void**)&this->_key, sizeof(T)*nKeys));
        CUDA_ERR(cudaMalloc((void**)&this->_value, sizeof(int)*nKeys));
    }

    T* _key;
    int* _value;
};

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CUDAKNN : public KNN<T>
{
    public:
        inline CUDAKNN(int dim, T* data, size_t bytes_size, enum mem_scope_t ptr_scope) 
            : KNN<T>(dim, data, bytes_size), _ptr_scope(ptr_scope)
        {
            if(_ptr_scope == PTR_TO_HOST_MEM)
            {
                CUDA_ERR(cudaMalloc((void**)&this->_data, this->_bytes_size));
                CUDA_ERR(cudaMemcpy(this->_data, data, this->_bytes_size, cudaMemcpyHostToDevice));
            }
        }

        inline ~CUDAKNN() 
        {
            if(_ptr_scope == PTR_TO_HOST_MEM) {
                CUDA_ERR(cudaFree(this->_data));
            }
        }

    protected:
        enum mem_scope_t _ptr_scope;
};

////////////////////////////////////////////////////////////////////////////////////////

template class CUDAKNN<float>;

////////////////////////////////////////////////////////////////////////////////////////
// Kernel Wrapper
/**
 * Number of threads must be equal to the number of points 
 * */
template <typename T>
extern void comp_dist(int blockPerGrid, int threadsPerBlock, T* dev_data, int dim, 
        int query, struct sort_t<T> dev_sort, int dev_data_bytes_size);

/**
 * Number of threads must be equal to the number of points * dim
 * */
template <typename T>
__host__ void comp_dist_opt(int blockPerGrid, int threadsPerBlock, T* dev_data, int dim, 
        int query, struct sort_t<T> dev_sort, int dev_data_bytes_size);

template <typename T>
extern void get_dim_comp_dist(int N_threads, int &blockPerGrid, int &threadsPerBlock);

template <typename T>
__host__ void get_dim_comp_dist_opt(int N_threads, int &blocksPerGrid, int &threadsPerBlock);

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////

