#include "matrix.hh"
#include "nn_exception.hh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{ }


void Matrix::allocateCudaMemory() {
	// don't actually want to allocate anything!
	// really just want to fetch the device pointer

	//if (!device_allocated) {
	//	float* device_memory = nullptr;
	//	cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
	//	NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
	//	data_device = std::shared_ptr<float>(device_memory,
	//										 [&](float* ptr){ cudaFree(ptr); });
	//	device_allocated = true;
	//}
	
	// sort of a redundant check, but just to be sure
	// since we're using copy memory now, we need to have the memory allocated, then 
	// "allocateCudaMemory" just gets the corresponding device pointer
	if (!host_allocated) {
		allocateHostMemory();
	}
	if (!device_allocated) {
		float* ptr;
		cudaHostGetDevicePointer(&ptr, data_host.get(), 0);
		
		// don't need to "free" the device pointer
		// may break when we try to free the shared pointer?
		//DON'T NEED A DELETER! its just the corresponding device pointer
		data_device = std::shared_ptr<float>(ptr, [&](float* ptr){});
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		//data_host = std::shared_ptr<float>(new float[shape.x * shape.y], [&](float* ptr){ delete[] ptr; });
		// temp value to hold the pointer
		float* ptr;
		// allocate like normal
		cudaHostAlloc(&ptr, shape.x * shape.y * sizeof(float), cudaHostAllocMapped);
		// make ptr a shared pointer, the "deleter" is a lambda function which calls
		// cudaFreeHost ???? i don't honestly know what the [&] is, something caputre list?
		data_host = std::shared_ptr<float>(ptr, [&](float* ptr) { cudaFreeHost(ptr); });
		host_allocated = true;
	}
}

void Matrix::allocateMemory() {
	// want to alloc host first, this was backward in original code
	allocateHostMemory();
	allocateCudaMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice() {
	// NO MORE COPYING!
	
	//if (device_allocated && host_allocated) {
	//	//cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
	//	//NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	//}
	//else {
	//	throw NNException("Cannot copy host data to not allocated memory on device.");
	//}
}

void Matrix::copyDeviceToHost() {
	// NO MORE COPYING!
	
	//if (device_allocated && host_allocated) {
	//	//cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
	//	//NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	//}
	//else {
	//	throw NNException("Cannot copy device data to not allocated memory on host.");
	//}
}

float& Matrix::operator[](const int index) {
	return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
	return data_host.get()[index];
}
