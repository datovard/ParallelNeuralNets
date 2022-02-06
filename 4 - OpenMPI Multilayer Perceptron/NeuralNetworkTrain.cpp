#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

// Training image file name
const string training_image_fn = "input/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "input/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "output/model-neural-network.dat";

// Report file name
const string report_fn = "output/training-report.dat";

// Number of training samples
const int nTraining = 60000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons
// iterations = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; // Ten classes: 0 - 9
const int iterations = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1, *delta1, *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2, *delta2, *in2, *out2, *theta2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
double *expected;

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. iterations: " << iterations << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
	cout << "Training label data: " << training_label_fn << endl;
	cout << "No. training sample: " << nTraining << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
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
}

// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

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

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+

void perceptron() {
  //#pragma omp parallel for
  for (int pos = 0; pos < n2; pos++) {
		in2[pos] = 0.0;
	}

  //#pragma omp parallel for
  for (int pos = 0; pos < n3; pos++) {
		in3[pos] = 0.0;
	}

  matrix_vector_multiplication(w1, out1, in2, n1, n2);

  //#pragma omp parallel for
  for (int pos = 0; pos < n2; pos++) {
		out2[pos] = sigmoid(in2[pos]);
	}

  matrix_vector_multiplication(w2, out2, in3, n2, n3);
    
  //#pragma omp parallel for
  for (int i = 0; i < n3; i++) {
		out3[i] = sigmoid(in3[i]);
	}
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error(){
  double res = 0.0;
  for (int i = 1; i <= n3; ++i) {
    res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
  }
  res *= 0.5;
  return res;
}

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+

void back_propagation() {
  double sum;

  //#pragma omp parallel for
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

// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+

int learning_process() {
  // #pragma omp parallel for
  for (int pos = 0; pos < n1 * n2; pos++) {
		delta1[pos] = 0.0;
	}

  //#pragma omp parallel for
  for (int pos = 0; pos < n2 * n3; pos++) {
		delta2[pos] = 0.0;
	}

  for (int i = 1; i <= iterations; ++i) {
    perceptron();
    back_propagation();
    if (square_error() < epsilon) {
      return i;
    }
  }
  return iterations;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

void input() {
  // Reading image
  char number;
  for (int j = 1; j <= height; ++j) {
    for (int i = 1; i <= width; ++i) {
      image.read(&number, sizeof(char));
      out1[i + (j - 1) * width] = (number == 0)? 0: 1;
    }
  }

  // Reading label
  label.read(&number, sizeof(char));
  for (int i = 1; i <= n3; ++i) {
    expected[i] = 0.0;
  }
  expected[number + 1] = 1.0;
}

// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name) {
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

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
  about();

  report.open(report_fn.c_str(), ios::out);
  image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
  label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

  // Reading file headers
  char number;
  for (int i = 1; i <= 16; ++i) {
    image.read(&number, sizeof(char));
  }
  for (int i = 1; i <= 8; ++i) {
    label.read(&number, sizeof(char));
  }

  // Neural Network Initialization
  init_array();

  for (int sample = 1; sample <= nTraining; ++sample) {
    // Getting (image, label)
    input();

    // Learning process: Perceptron (Forward procedure) - Back propagation
    int nIterations = learning_process();

    // Write down the squared error
    if(sample % 500 == 0){
      cout << "No. iterations: " << nIterations << endl;
      printf("Error: %0.6lf\n\n", square_error());
      report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;

      cout << "Saving the network to " << model_fn << " file." << endl;
      write_matrix(model_fn);
    }
  }

  // Save the final network
  write_matrix(model_fn);

  report.close();
  image.close();
  label.close();

  return 0;
}
