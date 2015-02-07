/*
 * =====================================================================================
 *       Filename:  cuda_knn_naive.hpp
 *    Description:  
 *        Created:  2015-02-02 21:04
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KNN_NAIVE_HPP
#define CUDA_KNN_NAIVE_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn.hpp"

#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct sort_t 
{
    T key;
    uint index;
};

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CUDAKNNNaive : public CUDAKNN<T>
{
    public:
        CUDAKNNNaive(uint dim, std::vector<T>& data);
        CUDAKNNNaive(uint dim, uint data_size_byte, T* _dev_data);
 
        /**
         * implementation of the method of the supper class knn  
         * */
        void find(uint query, uint k, std::vector<uint>& knn);

    protected:
        struct sort_t<T> * _dev_sort; // Array on device for sorting
        uint _dev_sort_byte;       // Size in bytes of the _dev_sort array
};

////////////////////////////////////////////////////////////////////////////////////////

template class CUDAKNNNaive<float>;

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
extern void comp_dist(uint blocksPerGrid, uint threadsPerBlock, T* dev_data, uint dim, 
        uint query, struct sort_t<T>* dev_sort);

template <typename T>
extern void comp_dist(uint N_threads, T* dev_data, uint dim, uint query, 
        struct sort_t<T>* dev_sort);

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_NAIVE_HPP */

////////////////////////////////////////////////////////////////////////////////////////

