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

#include <thrust/device_ptr.h>

#include "cuda_knn.hpp"
#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CUDAKNNThrust : public CUDAKNN<T>
{
    public:
        inline CUDAKNNThrust(uint dim, std::vector<T>& data) : 
            CUDAKNN<T>(dim, data), _dev_sort(this->_data.size() / this->_dim) {};
 
        /**
         * Implementation of the method of the supper class KNN  
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
        struct sort_t<T> _dev_sort;
};

////////////////////////////////////////////////////////////////////////////////////////

template class CUDAKNNThrust<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_THRUST_HPP */

////////////////////////////////////////////////////////////////////////////////////////

