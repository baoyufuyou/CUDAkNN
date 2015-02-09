/*
 * =====================================================================================
 *       Filename:  cpu_knn.hpp
 *    Description:  
 *        Created:  2015-02-02 11:59
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#ifndef CPU_KNN_HPP
#define CPU_KNN_HPP

////////////////////////////////////////////////////////////////////////////////////////

#include "knn.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class CPUKNN : public KNN<T>
{
    public:
        inline CPUKNN(uint dim, T* data, size_t bytes_size) 
            : KNN<T>(dim, data, bytes_size) {}

        /**
         * Implementation of the method of the supper class KNN  
         * */
        void find(uint query, uint k, std::vector<uint>& knn);

        /**
         * Compares a pair based on the first element
         * */
        inline static bool cmp_pair(const std::pair<T, uint> i, const std::pair<T, uint> j) 
        {
            return (i.first < j.first);
        }

    protected:
        /**
         * Evaluates the squared distance between a and b in the _data
         * using the norm 2
         * */
        T norm2_squared_dist(uint a, uint b);
};

////////////////////////////////////////////////////////////////////////////////////////

template class CPUKNN<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CPU_KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////

