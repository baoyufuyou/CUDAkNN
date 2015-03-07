/*
 * =====================================================================================
 *       Filename:  cuda_utils.cpp
 *    Description:  
 *        Created:  2015-02-08 15:39
 *         Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
 * =====================================================================================
 */

////////////////////////////////////////////////////////////////////////////////////////

#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "error.hpp"

////////////////////////////////////////////////////////////////////////////////////////

int CudaUtils::get_dev_free_mem()
{
    size_t free_mem, tot_mem;

    CUDA_ERR(cudaMemGetInfo(&free_mem, &tot_mem));

    return free_mem;
}

////////////////////////////////////////////////////////////////////////////////////////

int CudaUtils::get_dev_total_mem()
{
    size_t free_mem, tot_mem;

    CUDA_ERR(cudaMemGetInfo(&free_mem, &tot_mem));

    return tot_mem;
}

////////////////////////////////////////////////////////////////////////////////////////

int CudaUtils::get_dev_count()
{
    int dev_count;

    CUDA_ERR(cudaGetDeviceCount(&dev_count));

    return dev_count;
}

////////////////////////////////////////////////////////////////////////////////////////

#define PRINT_PROP(PROP) \
    std::cout << "  |----------------------------------------------------" << std::endl; \
    std::cout << "  |-> " << #PROP << ": " << prop.PROP << std::endl;
  

void CudaUtils::print_dev_info()
{
    int dev_count = CudaUtils::get_dev_count(); // Number of devices
    size_t total_mem_Mb = CudaUtils::get_dev_total_mem() / 1048576; // Total Memory in MB
    size_t free_mem_Mb  = CudaUtils::get_dev_free_mem() / 1048576;  // Free Memory in MB
    struct cudaDeviceProp prop;

    std::cout << "///////////////////////////////////////////////////////" << std::endl;
    std::cout << "// Memory: " << std::endl;
    std::cout << "// Free: " << free_mem_Mb << " MB | Total: " << total_mem_Mb << " MB" << std::endl;
 
    std::cout << "///////////////////////////////////////////////////////" << std::endl;
    std::cout << "// Number of devices: " << dev_count << std::endl;
    
    for(int i=0; i < dev_count; i++)
    {
        CUDA_ERR(cudaGetDeviceProperties(&prop, i));
        std::cout << "///////////////////////////////////////////////////////" << std::endl;
        std::cout << "|-> Name: " << prop.name << " " << i <<  std::endl;
        std::cout << "|__" << std::endl;

        PRINT_PROP(totalGlobalMem);
        PRINT_PROP(sharedMemPerBlock);
        PRINT_PROP(regsPerBlock);
        PRINT_PROP(warpSize);
        PRINT_PROP(memPitch);
        PRINT_PROP(maxThreadsPerBlock);
        PRINT_PROP(maxThreadsDim[3]);
        PRINT_PROP(maxGridSize[3]);
        PRINT_PROP(clockRate);
        PRINT_PROP(totalConstMem);
        PRINT_PROP(major);
        PRINT_PROP(minor);
        PRINT_PROP(textureAlignment);
        PRINT_PROP(texturePitchAlignment);
        PRINT_PROP(deviceOverlap);
        PRINT_PROP(multiProcessorCount);
        PRINT_PROP(kernelExecTimeoutEnabled);
        PRINT_PROP(integrated);
        PRINT_PROP(canMapHostMemory);
        PRINT_PROP(computeMode);
        PRINT_PROP(maxTexture1D);
        PRINT_PROP(maxTexture1DLinear);
        PRINT_PROP(maxTexture2D[2]);
        PRINT_PROP(maxTexture2DLinear[3]);
        PRINT_PROP(maxTexture2DGather[2]);
        PRINT_PROP(maxTexture3D[3]);
        PRINT_PROP(maxTextureCubemap);
        PRINT_PROP(maxTexture1DLayered[2]);
        PRINT_PROP(maxTexture2DLayered[3]);
        PRINT_PROP(maxTextureCubemapLayered[2]);
        PRINT_PROP(maxSurface1D);
        PRINT_PROP(maxSurface2D[2]);
        PRINT_PROP(maxSurface3D[3]);
        PRINT_PROP(maxSurface1DLayered[2]);
        PRINT_PROP(maxSurface2DLayered[3]);
        PRINT_PROP(maxSurfaceCubemap);
        PRINT_PROP(maxSurfaceCubemapLayered[2]);
        PRINT_PROP(surfaceAlignment);
        PRINT_PROP(concurrentKernels);
        PRINT_PROP(ECCEnabled);
        PRINT_PROP(pciBusID);
        PRINT_PROP(pciDeviceID);
        PRINT_PROP(pciDomainID);
        PRINT_PROP(tccDriver);
        PRINT_PROP(asyncEngineCount);
        PRINT_PROP(unifiedAddressing);
        PRINT_PROP(memoryClockRate);
        PRINT_PROP(memoryBusWidth);
        PRINT_PROP(l2CacheSize);
        PRINT_PROP(maxThreadsPerMultiProcessor);
    }
    
    std::cout << "///////////////////////////////////////////////////////" << std::endl;
    std::cout << std::endl;


}

////////////////////////////////////////////////////////////////////////////////////////
