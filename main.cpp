/*
 * =====================================================================================
 *       Filename:  main.cpp
 *    Description:  This file contains some basic calls to other functions and
 *    objects defined in the other files of this code
 *        Created:  2015-02-02 10:35
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_kd_tree.hpp"
#include "cuda_knn_naive.hpp"
#include "cpu_knn.hpp"
#include "error.hpp"
#include "utils.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void print_data(uint dim, const std::vector<T>& data);

inline void print_neighbors(const std::vector<uint>& neighbors);

inline void print_info();

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Contains some basic call to other function of the program
 * */
int main(int argc, char* argv[])
{
    uint n_points = (argc > 1 ? std::stoul(argv[1]) : 2); // number of points
    uint dim = (argc > 2 ? std::stoul(argv[2]) : 1);      // number of dimentions of the space
    uint k = (argc > 3 ? std::stoul(argv[3]) : 1);        // k neighbors to find
    uint query = (argc > 4 ? std::stoul(argv[4]) : 0);    // index of the query
 
    std::vector<uint> neighbors; // final answer (k-nearest neighbors from query)

    print_info();   // print program's info

    std::vector<float>& data = gen_random_data<float>(n_points, dim); // generates random data

    print_data(dim, data); // prints generated data

    /** 
     * CPU implementation starts here
     * */
    TIME_BETWEEN(
    CPUKNN<float> knn_cpu(dim, data); 

    neighbors.clear();
    knn_cpu.find(query, k, neighbors);
    );

    print_neighbors(neighbors); // print found neighbors
    /** 
     * CPU implementation finishes here 
     * */

    /**
     * GPU implementation starts here  
     * */
    TIME_BETWEEN(
    CUDAKNNNaive<float> knn_cuda_naive(dim, data);

    neighbors.clear();
    knn_cuda_naive.find(query, k, neighbors);
    );
    print_neighbors(neighbors); // print found neighbors
    /**
     * GPU implementation starts here  
     * */

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void print_data(uint dim, const std::vector<T>& data)
{
    std::cout << std::endl << "Data:" << std::endl;
    
    for(uint i=0; i < data.size() && i < 20; i+=dim)
    {
        std::cout << "(";
        for(uint j=0; j < dim-1; j++)
        {
            std::cout << data[i+j] << ",";
        }
        std::cout << data[i+dim-1] << ") ";
    }
    if(data.size() > 20)
        std::cout << "...";

    std::cout << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////

inline void print_neighbors(const std::vector<uint>& neighbors)
{ 
    std::cout << std::endl << "Neighbors:" << std::endl;

    if(neighbors.size() > 0){
        for(uint i=0; i < neighbors.size()-1 && i < 20; i++)
            std::cout << neighbors[i] << ",";
        std::cout << neighbors[neighbors.size()-1];

        if(neighbors.size() > 20)
            std::cout << "...";
    }
    
    std::cout << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////

inline void print_info()
{
    std::cout << 
        "CUDA KNN - A simple implementation of k-nearest neighbors in CUDA "    << std::endl << 
                                                                                   std::endl <<
        "Usage: cudaknn [n_points] [dim] [k] [query]"                           << std::endl <<
        "Options:"                                                              << std::endl <<
        "   n_points: number of random points to generate"                      << std::endl <<
        "   dim:      number of dimentions of each point"                       << std::endl << 
        "   k:        number of nearest neighbors to find"                      << std::endl << 
        "   query:    index of the query point (must be smaller than n_points)" << std::endl <<
                                                                                   std::endl;

}

////////////////////////////////////////////////////////////////////////////////////////
