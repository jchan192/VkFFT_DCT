FFTW_DIR=/cluster/gcc-11.2.0/fftw-3.3.10/

VkFFT: /datagpu04-5/jchan/VkFFT/DCT_Test/Vulkan_FFT.cu
	/cluster/cuda-11.6/bin/nvcc \
	-DVKFFT_BACKEND=1 -DVK_API_VERSION=11 \
	-I/datagpu04-5/jchan/VkFFT/vkFFT \
	-I/datagpu04-5/jchan/VkFFT/benchmark_scripts/vkFFT_scripts/include \
	-I/datagpu04-5/jchan/VkFFT/half_lib \
	-I${FFTW_DIR}/include  -L${FFTW_DIR}/lib -lfftw3 \
	-O3  -std=c++17 -x cu /datagpu04-5/jchan/VkFFT/DCT_Test/Vulkan_FFT.cu -o Vulkan_FFT.o -lcuda -lnvrtc 

