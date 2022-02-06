//%cflags:-fopenmp -lm -D_DEFAULT_SOURCE
#include "NeuralNetwork.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>
#include <sys/time.h>

using namespace std;

void matrix_vector_multiplication(const double* weights, const double* outputs, double* inputs, const int height, const int width){
    //#pragma omp parallel for
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            inputs[j] += outputs[i] * weights[i * width + j];
		}
	}
}

void adjust_weights(const double* thetas, const double* outputs, double* deltas, double* weights, const int height, const int width, const double learning_rate, const double momentum){
    //#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            deltas[i * width + j] = (learning_rate * thetas[j] * outputs[i]) + (momentum * deltas[i * width + j]);
            weights[i * width + j] += deltas[i * width + j];
        }
	}
}

// constructor of neural network class
NeuralNetwork::NeuralNetwork(int input_size, int label_size, int num_threads)
{
    omp_set_num_threads(num_threads);

    this->width = this->height = (input_size);
    this->n1 = input_size;
    this->n3 = label_size;

    // Layer 1 - Layer 2 = Input layer - Hidden layer
    w1 = new double[n1*n2];
    delta1 = new double[n1*n2];
    out1 = new double[n1];

	// Layer 2 - Layer 3 = Hidden layer - Output layer
    w2 = new double[n2*n3];
    delta2 = new double[n2 * n3];
    in2 = new double [n2];
    out2 = new double [n2];
    theta2 = new double [n2];

	// Layer 3 - Output layer
    in3 = new double [n3];
    out3 = new double [n3];
    theta3 = new double [n3];

    expected = new double [n3];
    
    // Initialization for weights from Input layer to Hidden layer
    #pragma omp parallel for
    for (int i = 0; i < n1 * n2; i++) {
        w1[i] = ((rand() % 2 == 1)? -1: 1) * (double)(rand() % 6) / 10.0;
	}
	
	// Initialization for weights from Hidden layer to Output layer
    #pragma omp parallel for
    for (int i = 0; i < n2 * n3; i++) {
        w2[i] = ((rand() % 2 == 1)? -1: 1) * (double)(rand() % 10 + 1) / (10.0 * n3);
	}
};

double activationFunction(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void NeuralNetwork::propagateForward()
{
    #pragma omp parallel for
    for (int pos = 0; pos < n2; pos++) {
		in2[pos] = 0.0;
	}

    #pragma omp parallel for
    for (int pos = 0; pos < n3; pos++) {
		in3[pos] = 0.0;
	}

    matrix_vector_multiplication(w1, out1, in2, n1, n2);

    #pragma omp parallel for
    for (int pos = 0; pos < n2; pos++) {
		out2[pos] = activationFunction(in2[pos]);
	}

    matrix_vector_multiplication(w2, out2, in3, n2, n3);
    
    #pragma omp parallel for
    for (int i = 0; i < n3; i++) {
		out3[i] = activationFunction(in3[i]);
	}
}

double NeuralNetwork::calculateErrors()
{
    double res = 0.0;
    for (int i = 0; i < n3; i++) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

void NeuralNetwork::propagateBackward()
{
    double sum;

    #pragma omp parallel for
    for (int pos = 0; pos < n3; pos++) {
        theta3[pos] = out3[pos] * (1 - out3[pos]) * (expected[pos] - out3[pos]);
	}
    
    //#pragma omp parallel for
    for (int i = 0; i < n2; i++) {
        sum = 0.0;
        for (int j = 0; j < n3; j++) {
            sum += w2[i * n3 + j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    adjust_weights(theta3, out2, delta2, w2, n2, n3, learning_rate, momentum);

    adjust_weights(theta2, out1, delta1, w1, n1, n2, learning_rate, momentum);
}

int NeuralNetwork::train()
{
    #pragma omp parallel for
    for (int pos = 0; pos < n1 * n2; pos++) {
		delta1[pos] = 0.0;
	}

    #pragma omp parallel for
    for (int pos = 0; pos < n2 * n3; pos++) {
		delta2[pos] = 0.0;
	}

    for (int i = 1; i <= epochs; ++i) {
        this->propagateForward();
        this->propagateBackward();
        if (this->calculateErrors() < epsilon) {
			return i;
		}
    }
    return epochs;
}

void NeuralNetwork::writeMatrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int pos = 0; pos < n1 * n2; pos++) {
		file << w1[pos] << " ";
		
        if( pos % width == n2 )
		    file << endl;
    }
	
	// Hidden layer - Output layer
    for (int pos = 0; pos < n2 * n3; pos++) {
		file << w2[pos] << " ";
		
        if( pos % width == n3 )
            file << endl;
    }
	
	file.close();
}

void NeuralNetwork::loadModel(string file_name) {
	ifstream file(file_name.c_str(), ios::in);
	
	// Input layer - Hidden layer
    for (int pos = 0; pos < n1 * n2; pos++) {
		file >> w1[pos];
    }
	
	// Hidden layer - Output layer
    for (int pos = 0; pos < n2 * n3; pos++) {
		file >> w2[pos];
    }
	
	file.close();
}

void NeuralNetwork::updateInput(bool* referenceInput){
    for(int position = 0; position < n1; position++){
        out1[position] = (double) referenceInput[position];
    }
}

void NeuralNetwork::updateExpectedOutput(bool* referenceOutput){
    for(int position = 0; position < n3; position++){
        expected[position] = (double) referenceOutput[position];
    }
}

double* NeuralNetwork::getOutput(){
    return this->out3;
}