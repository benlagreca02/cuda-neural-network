#include "bce_cost.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

#define BLOCK_SIZE 256

__global__ void binaryCrossEntropyCost(float* predictions, float* target,
									   int size, float* cost) {
	// the index within the block
	int tid = threadIdx.x;
	// the "real" index within the grid
	int idx = blockIdx.x * blockDim.x + tid;

	__shared__ float partialCosts[BLOCK_SIZE];
	partialCosts[tid] = 0;
	if (idx < size) {
		// load all calculated partial sum values to shared memory block
		partialCosts[tid] = target[idx] * logf(predictions[idx]) + (1.0f - target[idx]) * logf(1.0f - predictions[idx]);
		__syncthreads();

		// Reduction tree
		for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
			if(tid < stride){
				partialCosts[tid] += partialCosts[tid + stride];
			}
			__syncthreads();
		}
	}
	
	// get each block's value and add it up
	if (tid == 0) {
		atomicAdd(cost, -partialCosts[tid] / size);
	}
	
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY,
								     	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
	}
}

float BCECost::cost(Matrix predictions, Matrix target) {
	assert(predictions.shape.x == target.shape.x);

	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;

	dim3 block_size(BLOCK_SIZE);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														  target.data_device.get(),
														  predictions.shape.x, cost);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");

	float cost_value = *cost;
	cudaFree(cost);

	return cost_value;
}

Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														   target.data_device.get(),
														   dY.data_device.get(),
														   predictions.shape.x);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");

	return dY;
}
