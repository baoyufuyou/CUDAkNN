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

#include "knn.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CUDAKNN : public KNN<T>
{
    public:
        CUDAKNN(uint dim, std::vector<T>& data);
        CUDAKNN(uint dim, uint data_size_byte, T* _dev_data);

    protected:
        T* _dev_data; // data vector in device memory;
        uint _data_size_byte; // size of data in bytes loaded into device
};

////////////////////////////////////////////////////////////////////////////////////////

template class CUDAKNN<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////

