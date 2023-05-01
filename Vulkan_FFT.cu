#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include "vkFFT.h"
#include "utils_VkFFT.h"
#include "half.hpp"
#include "user_benchmark_VkFFT.h"

#include "sample_17_precision_VkFFT_double_dct.h"

//#ifdef USE_FFTW
#include "fftw3.h"
//#endif

VkFFTResult performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, int inverse, uint64_t num_iter) {
	VkFFTResult resFFT = VKFFT_SUCCESS;

	cudaError_t res = cudaSuccess;
	std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < num_iter; i++) {
		resFFT = VkFFTAppend(app, inverse, launchParams);
		if (resFFT != VKFFT_SUCCESS) return resFFT;
	}
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;

	return resFFT;

}
VkFFTResult transferDataToCPU(VkGPU* vkGPU, void* cpu_arr, void* output_buffer, uint64_t transferSize) {
	//a function that transfers data from the GPU to the CPU using staging buffer, because the GPU memory is not host-coherent
	VkFFTResult resFFT = VKFFT_SUCCESS;
	cudaError_t res = cudaSuccess;
	void* buffer = ((void**)output_buffer)[0];
	res = cudaMemcpy(cpu_arr, buffer, transferSize, cudaMemcpyDeviceToHost);
	if (res != cudaSuccess) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
	return resFFT;
}
VkFFTResult transferDataFromCPU(VkGPU* vkGPU, void* cpu_arr, void* input_buffer, uint64_t transferSize) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
	cudaError_t res = cudaSuccess;
	void* buffer = ((void**)input_buffer)[0];
	res = cudaMemcpy(buffer, cpu_arr, transferSize, cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		return VKFFT_ERROR_FAILED_TO_COPY;
	}
	return resFFT;
}

int main(int argc, char* argv[])
{

  int N = 16;
  VkGPU vkGPU = {};
  bool file_output = false;
  FILE* output = NULL;
  int sscanf_res = 0;
  double* inputC;
  inputC = (double*)(malloc(sizeof(double) * N));
  
  // initialize input
  for (int i=0; i<N;i++){
    inputC[i] = sin(i * 2.*3.14/N);
  }

  VkFFTResult resFFT = VKFFT_SUCCESS;
  VkFFTConfiguration configuration = {};
  VkFFTApplication app = {};
  
  // initialize device 
  CUresult res = CUDA_SUCCESS;
  cudaError_t res2 = cudaSuccess;
  res = cuInit(0);
  res2 = cudaSetDevice( (int) (vkGPU.device_id));
  if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
  res = cuDeviceGet(&vkGPU.device,(int)  (vkGPU.device_id));
  res = cuCtxCreate(&vkGPU.context, 0,  vkGPU.device);
  res = cuCtxDestroy(vkGPU.context);


  // initialize configuration
  configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
  configuration.size[0] = N; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
  	//	configuration.size[1] = benchmark_dimensions[n][1];
  	//	configuration.size[2] = benchmark_dimensions[n][2];
  configuration.performDCT = 1;// DCT type. Currently supported 2, 3 and 4
  configuration.doublePrecision = 1;
  configuration.device = &vkGPU.device;

  uint64_t numBuf = 1;
  uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
  bufferSize[0] = {};
  bufferSize[0] = (uint64_t)sizeof(double) * configuration.size[0] / numBuf;

  cuFloatComplex* buffer = 0;
  configuration.bufferNum = numBuf;
  configuration.bufferSize = bufferSize;
  res2 = cudaMalloc((void**)&buffer, bufferSize[0]);
  
  // initialize VkFFT
  uint64_t shift = 0;
  resFFT = transferDataFromCPU(&vkGPU, (inputC), &buffer, bufferSize[0]);
  //  resFFT = transferDataFromCPU(&vkGPU, (inputC + shift / sizeof(double)), &buffer, bufferSize[0]);
  resFFT = initializeVkFFT(&app, configuration);
  if (resFFT == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_DCT) {
    printf("VkFFT DCT-%d System: unsupported !! \n");
  }

  // cudaFree(buffer);
  // free(bufferSize);
  // deleteVkFFT(&app);
  // free(inputC);
					
  // perform VkFFT
  int num_iter = 1;
  VkFFTLaunchParams launchParams = {};
  launchParams.buffer = (void**)&buffer;
  resFFT = performVulkanFFT(&vkGPU, &app, &launchParams, -1, num_iter);
  double* output_VkFFT = (double*)(malloc(sizeof(double) * N));

  shift = 0;
  resFFT = transferDataToCPU(&vkGPU, (output_VkFFT), &buffer, bufferSize[0]);

  // Perform FFTW3 for comparison
  double* inputC_double = (double*)(malloc(sizeof(double) * N));
  double* output_FFTW = (double*)(malloc(sizeof(double) * N));
  fftw_plan p;
  for (int i=0; i<N;i++){
    inputC_double[i] = sin(i * 2.*3.14/N);
  }
  p = fftw_plan_r2r_1d(N, inputC_double, output_FFTW, FFTW_REDFT00, FFTW_ESTIMATE);
  fftw_execute(p);

  // printer  output
  for (int i=0; i<N;i++)
    std::cout << i * 2.*3.14/N << " " << inputC[i] << " " << output_VkFFT[i] << " " << output_FFTW[i] <<"\n";  


  return 0;
}
