/*
 * =====================================================================================
 *       Filename:  cuda_knn_thrust.hpp
 *    Description:  
 *        Created:  2015-02-09 10:26
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KNN_THRUST_HPP
#define CUDA_KNN_THRUST_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn.hpp"
#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CUDAKNNThrust : public CUDAKNN<T>
{
    public:
        inline CUDAKNNThrust(uint dim, T* data, size_t bytes_size, enum mem_scope_t ptr_scope) : 
            CUDAKNN<T>(dim, data, bytes_size, ptr_scope), 
            _dev_sort(this->_bytes_size / (this->_dim * sizeof(T))) {};
 
        inline ~CUDAKNNThrust() { this->free(); }

        /**
         * Implementation of the method of the supper class KNN  
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
        struct sort_t<T> _dev_sort;
};

////////////////////////////////////////////////////////////////////////////////////////

template class CUDAKNNThrust<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_THRUST_HPP */

////////////////////////////////////////////////////////////////////////////////////////

