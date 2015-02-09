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
#include "parser.hpp"
#include "cuda_knn_thrust.hpp"

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void print_data(uint dim, const std::vector<T>& data);

inline void print_neighbors(const std::vector<uint>& neighbors);

template <typename T>
inline void print_keys(const T* keys, uint size);

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Contains some basic call to other function of the program
 * */
int main(int argc, char* argv[])
{
    uint n_points; // number of points
    uint dim;      // number of dimentions of the space
    uint k;        // k neighbors to find
    uint query;    // index of the query
    
    Parser parser(argc, argv);
    parser.get_options(n_points, dim, k, query);

    std::cout << "Using: " << std::endl;
    std::cout << "n_points: " << n_points <<  "| dim: " << dim << "| k: " << k << "| query: " << query << std::endl;

    std::vector<uint> neighbors; // final answer (k-nearest neighbors from query)

    std::vector<float>& data = gen_random_data<float>(n_points, dim); // generates random data

    print_data(dim, data); // prints generated data

    /** 
     * CPU implementation starts here
     * */
    std::cout << "=== CPU Implementation ===================================" << std::endl;
    TIME_BETWEEN(
    CPUKNN<float> knn_cpu(dim, data); 

    neighbors.clear();
    knn_cpu.find(query, k, neighbors);
    );
    print_neighbors(neighbors); // print found neighbors
    std::cout << "==========================================================" << std::endl << std::endl;
    /** 
     * CPU implementation finishes here 
     * */

    /**
     * GPU naive implementation starts here  
     * */
    std::cout << "=== GPU Implementation Naive =============================" << std::endl;
    TIME_BETWEEN(
    CUDAKNNNaive<float> knn_cuda_naive(dim, data);

    neighbors.clear();
    knn_cuda_naive.find(query, k, neighbors);
    );
    print_neighbors(neighbors); // print found neighbors
    std::cout << "==========================================================" << std::endl << std::endl;
    /**
     * GPU naive implementation ends here  
     * */

    /**
     * GPU thrust implementation starts here  
     * */
    std::cout << "=== GPU Implementation Thrust ============================" << std::endl;
    TIME_BETWEEN(
    CUDAKNNThrust<float> knn_cuda_thrust(dim, data);

    neighbors.clear();
    knn_cuda_thrust.find(query, k, neighbors);
    );
    print_neighbors(neighbors); // print found neighbors
    std::cout << "==========================================================" << std::endl << std::endl;
    /**
     * GPU thrust implementation ends here  
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
    std::cout << "Neighbors:" << std::endl;

    if(neighbors.size() > 0){
        for(uint i=0; i < neighbors.size()-1 && i < 20; i++)
            std::cout << neighbors[i] << ",";
        std::cout << neighbors[neighbors.size()-1];

        if(neighbors.size() > 20)
            std::cout << "...";
    }
    
    std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void print_keys(const T* keys, uint size)
{
    T* keys_host = new T[size];

    CUDA_ERR(cudaMemcpy(keys_host, keys, sizeof(T)*size, cudaMemcpyDeviceToHost));

    for(uint i=0; i < size-1 && i < 20; i++)
    {
        std::cout << keys_host[i] << ",";
    }
    std::cout << keys_host[size-1];

    if(size > 20)
        std::cout << "...";

    std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////
