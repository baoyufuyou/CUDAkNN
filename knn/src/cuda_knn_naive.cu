/*
 * =====================================================================================
 *       Filename:  cuda_knn_naive.cpp
 *    Description:
 *        Created:  2015-02-03 19:18
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_knn_naive.hpp"

#include <cuda_runtime.h>

#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void bitonic_sort_step(struct sort_t<T> dev_values, int j, int k, int key_length)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;

	/* The threads with the lowest ids sort the array. */
	if(i < key_length && ixj < key_length)
	{
		if ((ixj)>i) {
			if ((i&k)==0) {
				// Sort ascending
				if (dev_values._key[i] > dev_values._key[ixj]) {
					// exchange(i,ixj);
					T temp = dev_values._key[i];
					int temp2 = dev_values._value[i];

					dev_values._key[i] = dev_values._key[ixj];
					dev_values._value[i] = dev_values._value[ixj];

					dev_values._key[ixj] = temp;
					dev_values._value[ixj] = temp2;
				}
			}
			if ((i&k)!=0) {
				// Sort descending
				if (dev_values._key[i] < dev_values._key[ixj]) {
					// exchange(i,ixj);
					T temp = dev_values._key[i];
					int temp2 = dev_values._value[i];

					dev_values._key[i] = dev_values._key[ixj];
					dev_values._value[i] = dev_values._value[ixj];

					dev_values._key[ixj] = temp;
					dev_values._value[ixj] = temp2;
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Inplace bitonic sort using CUDA.
 */
template <typename T>
__host__
void bitonic_sort(struct sort_t<T>& values, int blockPerGrid, int threadsPerBlock, int key_length)
{
	int j, k;
	int max_val = blockPerGrid * threadsPerBlock;

	/* Major step */
	for (k = 2; k <= max_val; k <<= 1) {
		/* Minor step */
		for (j=k>>1; j>0; j=j>>1) {
			bitonic_sort_step<T><<<blockPerGrid, threadsPerBlock>>>(values, j, k, key_length);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ void get_dim_bitonic_sort_opt(int N_threads, int &blocksPerGrid, int &threadsPerBlock)
{
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock,
            (void*)bitonic_sort_step<T>, 0, N_threads);

    blocksPerGrid = (N_threads + threadsPerBlock - 1) / threadsPerBlock;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ void CUDAKNNNaive<T>::find(int query, int k, std::vector<int>& knn)
{
	int N_threads = this->_bytes_size / (this->_dim * sizeof(T));
	int blockPerGrid, threadsPerBlock;

	query = query * this->_dim;

	get_dim_comp_dist<T>(N_threads, blockPerGrid, threadsPerBlock);
	comp_dist<T>(blockPerGrid, threadsPerBlock, this->_data, this->_dim, query,
			this->_dev_sort, this->_bytes_size);

	get_dim_bitonic_sort_opt<T>(N_threads, blockPerGrid, threadsPerBlock);
	bitonic_sort<T>(this->_dev_sort, blockPerGrid, threadsPerBlock, this->_bytes_size / this->_dim / sizeof(T));

	knn.resize(k);
	CUDA_ERR(cudaMemcpy(knn.data(), this->_dev_sort._value + 1, k*sizeof(int),
			cudaMemcpyDeviceToHost));
}

////////////////////////////////////////////////////////////////////////////////////////
