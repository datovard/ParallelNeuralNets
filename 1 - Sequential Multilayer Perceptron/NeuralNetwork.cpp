#include "NeuralNetwork.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

double activationFunction(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// constructor of neural network class
NeuralNetwork::NeuralNetwork(int input_size, int label_size)
{
    this->width = this->height = (input_size);
    this->n1 = input_size;
    this->n3 = label_size;

    // Layer 1 - Layer 2 = Input layer - Hidden layer
    this->w1 = (double **) malloc( (n1 + 1) * sizeof(double*) );
    this->delta1 = (double **) malloc( (n1 + 1) * sizeof(double*) );

    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double [n2 + 1];
        delta1[i] = new double [n2 + 1];
    }

    out1 = new double [n1 + 1];

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    this->w2 = (double **) malloc( (n2 + 1) * sizeof(double*) );
    this->delta2 = (double **) malloc( (n2 + 1) * sizeof(double*) );

    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double [n3 + 1];
        delta2[i] = new double [n3 + 1];
    }

    in2 = new double [n2 + 1];
    out2 = new double [n2 + 1];
    theta2 = new double [n2 + 1];

    // Layer 3 - Output layer
    in3 = new double [n3 + 1];
    out3 = new double [n3 + 1];
    theta3 = new double [n3 + 1];

    expected = new double [n3 + 1];

    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            int sign = rand() % 2;

            // Another strategy to randomize the weights - quite good 
            // w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);
            
            w1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
				w1[i][j] = - w1[i][j];
			}
        }
	}

    // Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            int sign = rand() % 2;
			
			// Another strategy to randomize the weights - quite good 
            // w2[i][j] = (double)(rand() % 6) / 10.0;

            w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
				w2[i][j] = - w2[i][j];
			}
        }
	}
};


void NeuralNetwork::propagateForward()
{
    for (int i = 1; i <= n2; ++i) {
		in2[i] = 0.0;
	}

    for (int i = 1; i <= n3; ++i) {
		in3[i] = 0.0;
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            in2[j] += out1[i] * w1[i][j];
		}
	}

    for (int i = 1; i <= n2; ++i) {
		out2[i] = activationFunction(in2[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * w2[i][j];
		}
	}

    for (int i = 1; i <= n3; ++i) {
		out3[i] = activationFunction(in3[i]);
	}
}

void NeuralNetwork::propagateBackward()
{
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
	}
}

double NeuralNetwork::calculateErrors()
{
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

int NeuralNetwork::train()
{
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			delta1[i][j] = 0.0;
		}
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			delta2[i][j] = 0.0;
		}
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

// +------------------------+
// | Saving weights to file |
// +------------------------+

void NeuralNetwork::write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i = 1; i <= this->n1; ++i) {
        for (int j = 1; j <= this->n2; ++j) {
			file << this->w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i = 1; i <= this->n2; ++i) {
        for (int j = 1; j <= this->n3; ++j) {
			file << this->w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}

void NeuralNetwork::updateInput(bool* referenceInput){
    for(int position = 1; position <= n1; position++){
        out1[position] = (double) referenceInput[position-1];
    }
}

void NeuralNetwork::updateExpectedOutput(bool* referenceOutput){
    for(int position = 1; position <= n3; position++){
        expected[position] = (double) referenceOutput[position-1];
    }
}