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
        uint _dim; // Number of dimentions of the data
        std::vector<T>& _data; // Data
    
    public:
        /**
         * Constructor:
         * @arg1: dimention of the space that the data is defined
         * @arg2: array of the data 
         * */
        inline KNN(uint dim, std::vector<T>& data) : _dim(dim), _data(data) {
            ASSERT_FATAL_ERROR((_data.size() % _dim) == 0, \
                    "data size must devide number of dimentions");
        };

        /**
         * Resets the data and its dimention 
         * */
        inline void reset(uint dim, std::vector<T>& data);

        /**
         * Returns the k-nearest neighbors of query  
         * @arg1: index in the vector data of the query
         * @arg2: k neighbors to find
         * @arg3: vector that will return the indices of the k nearest neighbors
         * WARNING: the query index is not _data[query] but 
         * _data[query * dim], as well as for the return vector
         * */
        virtual void find(uint query, uint k, std::vector<uint>& knn);

        /**
         * Gets  
         * */
        inline const std::vector<T>& data() const {return _data;}
        inline uint dim() const {return _dim;}

        /**
         * Sets 
         * */
        inline std::vector<T>& data() {return _data;}
        inline uint& dim() {return _dim;}
};

////////////////////////////////////////////////////////////////////////////////////////

template class KNN<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////

