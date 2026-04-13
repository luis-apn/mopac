/**
 * @file gpu_info.cu
 * @author Carlos Peixoto M Junior and Julio Daniel Carvalho Maia
 * @date 11/14/2013
 */
 
/*****************************************************************************/
/*                            INCLUDES                                       */
/*****************************************************************************/
#include <cuda.h>
#include <stdio.h>
/**
 *
 */
extern "C" void getGPUInfo(bool *hasGpu, bool *hasDouble, int *nDevices, char *name, int *name_size, 
						   size_t *totalMem, int *clockRate, int *major, int *minor)
{
    int n;
    cudaError_t error = cudaGetDeviceCount(&n);
    
    if(error == cudaErrorNoDevice)
    {
        *hasGpu    = false;
        //*hasDouble = false;
        *nDevices  = 0;
    }
    
    else if(error == cudaErrorInsufficientDriver)
    {
        *hasGpu    = false;
        //*hasDouble = false;
        *nDevices  = 0;
    }
    else
    {
        *hasGpu    = true;
        *nDevices  = n;
        cudaDeviceProp prop;
        

	for (int i = 0; i < n; i++){

		if(cudaGetDeviceProperties(&prop, i) != cudaErrorInvalidDevice)
		{
		    if(prop.major >= 2)
		        hasDouble[i] = true;
		    else
		        hasDouble[i] = false;
		}
						
		strcpy(name + (i*256), prop.name);
		name_size[i] = (int)strlen(name + (i*256));
		totalMem[i] = prop.totalGlobalMem;
		clockRate[i] = 0; // clockRate removed in CUDA 13.x
		minor[i] = prop.minor;
		major[i] = prop.major;
    
	}
}
}

// ============================================================================
// Set the active CUDA device for the current process
// Called from Fortran via settingGPUcard module (mod_gpu_info.F90)

extern "C" void setDevice(int idevice, bool *stat)
{
    cudaError_t error = cudaSetDevice(idevice);
    *stat = (error == cudaSuccess);
    if (error != cudaSuccess) {
        printf("Failed to set GPU device %d: %s\n", idevice, cudaGetErrorString(error));
    }
}
