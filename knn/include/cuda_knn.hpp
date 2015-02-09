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

#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct sort_t 
{
    inline sort_t(uint nKeys)
    {
        CUDA_ERR(cudaMalloc((void**)&_key, sizeof(T)*nKeys));
        CUDA_ERR(cudaMalloc((void**)&_value, sizeof(uint)*nKeys));
    }

    T* _key;
    uint* _value;
};

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

template <typename T>
CUDAKNN<T>::CUDAKNN(uint dim, std::vector<T>& data)
    :
        KNN<T>(dim, data),
        _dev_data(NULL)
{
    T* host_data = this->_data.data();

    _data_size_byte = this->_data.size() * sizeof(T);

    CUDA_ERR(cudaMalloc((void**)&_dev_data, _data_size_byte));
    CUDA_ERR(cudaMemcpy(_dev_data, host_data, _data_size_byte, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUDAKNN<T>::CUDAKNN(uint dim, uint data_size_byte, T* dev_data)
    :
        KNN<T>(dim, *(new std::vector<T>(0))),
        _dev_data(dev_data),
        _data_size_byte(data_size_byte)
{
    /* Nothing to do here */
}

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CUDA_KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////

