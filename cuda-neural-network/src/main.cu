#include <iostream>
#include <time.h>

#include <chrono>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"

#include "coordinates_dataset.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

// Very small for benchmarking purposes
const int NUM_EPOCHS = 100;
const int PRINT_FREQUENCY = 25;
int main() {

	srand( time(NULL) );

	CoordinatesDataset dataset(100, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));
	
	// To keep me from having to type "std::chrono:: " every time I need something
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	duration<double, std::milli> forwardTotalTimeMs(0);
	duration<double, std::milli> backwardTotalTimeMs(0);
	duration<double, std::milli> costCalcTotalTimeMs(0);
	

	// the result of forward prop, i.e. a 'guess'
	Matrix Y;

	printf("Training for %d epochs and printing every %d\n", NUM_EPOCHS, PRINT_FREQUENCY);

	// network training loop
	for (int epoch = 0; epoch < NUM_EPOCHS+1; epoch++) {
		float cost = 0.0;
		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			
			auto t0 = high_resolution_clock::now();
			Y = nn.forward(dataset.getBatches().at(batch));
			
			auto forwardTime = high_resolution_clock::now();
			nn.backprop(Y, dataset.getTargets().at(batch));
			auto backwardTime = high_resolution_clock::now();

			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
			auto costCalcTime = high_resolution_clock::now();

			// store timing data
			duration<double, std::milli> tmp = forwardTime - t0;
			forwardTotalTimeMs += tmp;
			
			tmp = backwardTime - forwardTime;
			backwardTotalTimeMs += tmp;

			tmp = costCalcTime - backwardTime;
			costCalcTotalTimeMs += tmp;
		}

		if (epoch % 25 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	// compute accuracy
	// not gonna time this one since its just once
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(
			Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;
	auto totalTimeMs = forwardTotalTimeMs + backwardTotalTimeMs + costCalcTotalTimeMs;

	printf("\n====== TIMINGS (ms) ====== \n");
	printf("Forward prop time: \t %12.4f \t %5.2f \n", forwardTotalTimeMs.count(), (forwardTotalTimeMs.count() / totalTimeMs.count()) * 100);
	printf("Backward prop time: \t %12.4f \t %5.2f \n", backwardTotalTimeMs.count(), (backwardTotalTimeMs.count() / totalTimeMs.count()) * 100);
	printf("Cost Calculation time: \t %12.4f \t %5.2f \n", costCalcTotalTimeMs.count(), (costCalcTotalTimeMs.count() / totalTimeMs.count()) * 100);
	printf("Total execution time: \t %12.4f\n", totalTimeMs.count());
	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
