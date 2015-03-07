/*
 * =====================================================================================
 *       Filename:  knn.hpp
 *    Description:  
 *        Created:  2015-02-02 12:00
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef KNN_HPP
#define KNN_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "utils.hpp"
#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class KNN 
{
    protected:
        int _dim; // Number of dimentions of the data
        T* _data; // Data
        size_t _bytes_size; // Size of the data
    
    public:
        /**
         * Constructor:
         * @arg1: dimention of the space that the data is defined
         * @arg2: array of the data 
         * */
            inline KNN(int dim, T* data, size_t bytes_size) : 
                _dim(dim), _data(data), _bytes_size(bytes_size) 
            {
                ASSERT_FATAL_ERROR((bytes_size % _dim) == 0, \
                    "data size must devide number of dimentions");
            };

        /**
         * Resets the data and its dimention 
         * */
        inline void reset(int dim, T* data, size_t bytes_size);

        /**
         * Returns the k-nearest neighbors of query  
         * @arg1: index in the vector data of the query
         * @arg2: k neighbors to find
         * @arg3: vector that will return the indices of the k nearest neighbors
         * WARNING: the query index is not _data[query] but 
         * _data[query * dim], as well as for the return vector
         * */
        virtual void find(int query, int k, std::vector<int>& knn);

        /**
         * Gets  
         * */
        inline T* data() const {return _data;}
        inline size_t bytes_size() const {return _bytes_size;}
        inline int dim() const {return _dim;}

        /**
         * Sets 
         * */
        inline T* data() {return _data;}
        inline size_t& bytes_size() {return _bytes_size;}
        inline int& dim() {return _dim;}
};

////////////////////////////////////////////////////////////////////////////////////////

template class KNN<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////

