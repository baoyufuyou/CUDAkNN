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

#include <cuda_runtime.h>

#include "cuda_knn.hpp"

#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CUDAKNNNaive : public CUDAKNN<T>
{
    public:
        inline CUDAKNNNaive(uint dim, T* data, size_t bytes_size, enum mem_scope_t ptr_scope) :
            CUDAKNN<T>(dim, data, bytes_size, ptr_scope),
            _dev_sort(this->_bytes_size / (this->_dim * sizeof(T))) {};
 
        inline ~CUDAKNNNaive() { this->free(); }

        /**
         * implementation of the method of the supper class knn  
         * */
        void find(uint query, uint k, std::vector<uint>& knn);

        /**
         * Frees the memory alloc in the device
         * */
        inline void free() {
            if(_dev_sort._key != NULL)
                CUDA_ERR(cudaFree(_dev_sort._key));  _dev_sort._key = NULL;
            if(_dev_sort._value != NULL)
                CUDA_ERR(cudaFree(_dev_sort._value));_dev_sort._value = NULL;
        }

        /**
         * Gets  
         * */
        inline const struct sort_t<T>& dev_sort() const {return _dev_sort;}

        /**
         * Sets  
         * */
        inline struct sort_t<T>& dev_sort() {return _dev_sort;}
    
    protected:
        struct sort_t<T> _dev_sort; // Array on device for sorting
};

////////////////////////////////////////////////////////////////////////////////////////

template class CUDAKNNNaive<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_NAIVE_HPP */

////////////////////////////////////////////////////////////////////////////////////////

