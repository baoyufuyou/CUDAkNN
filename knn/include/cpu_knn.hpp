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
        inline CPUKNN(uint dim, const std::vector<T> data) : KNN<T>(dim, data) {}

        /**
         * Implementation of the method of the supper class KNN  
         * */
        std::vector<T>& find(uint query, uint k);
};

////////////////////////////////////////////////////////////////////////////////////////

template class CPUKNN<float>;

////////////////////////////////////////////////////////////////////////////////////////

#endif /* !CPU_KNN_HPP */

////////////////////////////////////////////////////////////////////////////////////////
