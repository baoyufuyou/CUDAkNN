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
class CUDAKNNNaive : public CUDAKNN<T>
{
    public:
        CUDAKNNNaive(uint dim, std::vector<T>& data);
 
        /**
         * implementation of the method of the supper class knn  
         * */
        void find(uint query, uint k, std::vector<uint>& knn);

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

template <typename T>
CUDAKNNNaive<T>::CUDAKNNNaive(uint dim, std::vector<T>& data)
    :
        CUDAKNN<T>(dim, data), _dev_sort(this->_data.size()/this->_dim)
{
    /* Nothing to do here */
}

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_NAIVE_HPP */

////////////////////////////////////////////////////////////////////////////////////////

