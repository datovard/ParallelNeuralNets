// ejecutar con mpirun -np 4 mpi_test
// mpicc OpenMPI-Test.c -o OpenMPI-Test && mpirun -np 6 --hostfile mpi-hosts OpenMPI-Test

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include <mpi.h>
#include "FileReading.cpp"

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
const string report_train_fn = "output/training-report.dat";

// Report file name
const string report_test_fn = "output/testing-report.dat";

// Number of training samples
const int nTraining = 150;

// Number of testing samples
const int nTesting = 10000;

// Number of distributed hosts
const int nHosts = 4;

// Size of each batch on each host
const int batch_size = nTraining / (nHosts - 1);

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
const int epochs = 1;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1, *delta1, *out1;
double *original_w1;
double *final_w1;
double *difference_w1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2, *delta2, *in2, *out2, *theta2;
double *original_w2;
double *final_w2;
double *difference_w2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
double *expected;

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ofstream report_train;
ofstream report_test;

double **in_dat, **out_dat;
double **in_test, **out_test;

void copy_array(const double * original, double * copy, int size) {
  for (int i = 0; i < size; i++) {
    copy[i] = original[i];
	}
}

void clean_array(double * array, int size) {
  for (int i = 0; i < size; i++) {
    array[i] = 0;
	}
}

void add_cumulative_array(const double * original, double * cumulative, int size) {
  for (int i = 0; i < size; i++) {
    cumulative[i] += original[i];
	}
}

void get_weights_difference(double * original, double * weights, double * finals, int size, int model_amount){
  for(int i = 0; i < size; i++){
    double change = (original[i] - weights[i] <= 0)? (weights[i] - original[i]): -1*(original[i] - weights[i]);
    finals[i] += change / model_amount;
  }
}

void init_arrays() {
  // Layer 1 - Layer 2 = Input layer - Hidden layer
  w1 = new double[n1*n2];
  original_w1 = new double[n1*n2];
  difference_w1 = new double[n1*n2];
  final_w1 = new double[n1*n2];
  delta1 = new double[n1*n2];
  out1 = new double[n1];

  // Layer 2 - Layer 3 = Hidden layer - Output layer
  w2 = new double[n2*n3];
  original_w2 = new double[n2*n3];
  difference_w2 = new double[n2*n3];
  final_w2 = new double[n2*n3];
  delta2 = new double[n2 * n3];
  in2 = new double [n2];
  out2 = new double [n2];
  theta2 = new double [n2];

  // Layer 3 - Output layer
  in3 = new double [n3];
  out3 = new double [n3];
  theta3 = new double [n3];

  expected = new double [n3];
}

void fill_weights(){
  // Initialization for weights from Input layer to Hidden layer
  for (int i = 0; i < n1 * n2; i++) {
    original_w1[i] = ((rand() % 2 == 1)? -1: 1) * (double)(rand() % 6) / 10.0;
	}
	
	// Initialization for weights from Hidden layer to Output layer
  for (int i = 0; i < n2 * n3; i++) {
    original_w2[i] = ((rand() % 2 == 1)? -1: 1) * (double)(rand() % 10 + 1) / (10.0 * n3);
  }
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

void matrix_vector_multiplication(const double* weights, const double* outputs, double* inputs, const int height, const int width){
  #pragma omp parallel for
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < height; i++) {
      inputs[j] += outputs[i] * weights[i * width + j];
		}
	}
}

void adjust_weights(const double* thetas, const double* outputs, double* deltas, double* weights, const int height, const int width, const double learning_rate, const double momentum){
  #pragma omp parallel for
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      deltas[i * width + j] = (learning_rate * thetas[j] * outputs[i]) + (momentum * deltas[i * width + j]);
      weights[i * width + j] += deltas[i * width + j];
    }
	}
}

void perceptron() {
  clean_array(in2, n2);
  clean_array(in3, n3);

  matrix_vector_multiplication(w1, out1, in2, n1, n2);

  #pragma omp parallel for
  for (int pos = 0; pos < n2; pos++) {
		out2[pos] = sigmoid(in2[pos]);
	}

  matrix_vector_multiplication(w2, out2, in3, n2, n3);
    
  #pragma omp parallel for
  for (int i = 0; i < n3; i++) {
		out3[i] = sigmoid(in3[i]);
	}
}

void final_perceptron() {
  clean_array(in2, n2);
  clean_array(in3, n3);

  matrix_vector_multiplication(final_w1, out1, in2, n1, n2);

  #pragma omp parallel for
  for (int pos = 0; pos < n2; pos++) {
		out2[pos] = sigmoid(in2[pos]);
	}

  matrix_vector_multiplication(final_w2, out2, in3, n2, n3);
    
  #pragma omp parallel for
  for (int i = 0; i < n3; i++) {
		out3[i] = sigmoid(in3[i]);
	}
}

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

  #pragma omp parallel for
  for (int pos = 0; pos < n3; pos++) {
    theta3[pos] = out3[pos] * (1 - out3[pos]) * (expected[pos] - out3[pos]);
	}
    
  #pragma omp parallel for
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

int learning_process() {
  clean_array(delta1, n1 * n2);
  clean_array(delta2, n2 * n3);

  int i = 1;
  for (i; i <= iterations; ++i) {
    perceptron();
    back_propagation();
    if (square_error() < epsilon) {
      break;
    }
  }

  return i;
}

void input(const double* image, const double* label) {
  // Reading image
  for(int i = 0; i < height * width; i++){
    out1[i] = image[i];
  }

  // Reading label
  for (int i = 0; i < n3; i++) {
    expected[i] = label[i];
  }
}

void write_matrix(string file_name) {
  ofstream file(file_name.c_str(), ios::out);

  // Input layer - Hidden layer
  for (int pos = 0; pos < n1 * n2; pos++) {
		file << final_w1[pos] << " ";
		
    if( pos % width == n2 )
      file << endl;
  }
	
	// Hidden layer - Output layer
  for (int pos = 0; pos < n2 * n3; pos++) {
		file << final_w2[pos] << " ";
		
    if( pos % width == n3 )
      file << endl;
  }

  file.close();
}

void read_dataset(){
  report_train.open(report_train_fn.c_str(), ios::out);
  report_test.open(report_test_fn.c_str(), ios::out);
  in_dat = (double **) malloc( (nTraining+1) * sizeof(double*) );
  out_dat = (double **) malloc( (nTraining+1) * sizeof(double*) );
  in_test = (double **) malloc( (nTesting+1) * sizeof(double*) );
  out_test = (double **) malloc( (nTesting+1) * sizeof(double*) );

  ReadCSVOnDouble("input/training_input.csv", in_dat, nTraining);
  ReadCSVOnDouble("input/training_labels.csv", out_dat, nTraining);
  ReadCSVOnDouble("input/testing_input.csv", in_test, nTesting);
  ReadCSVOnDouble("input/testing_labels.csv", out_test, nTesting);
}

void test_model(int node){
  int nCorrect = 0, label = 0;;
  for (int sample = 0; sample < nTesting; sample++) {
    // Getting (image, label)
    input(in_test[sample], out_test[sample]);

    label = 0;
    for( label; label < 9; label++){
      if(out_test[sample][label]) break;
    }
  
    // Classification - Perceptron procedure
    final_perceptron();
      
    /// Prediction
    int predict = 0;
    for (int i = 1; i < n3; i++) {
      if (out3[i] > out3[predict]) {
        predict = i;
      }
    }

    // Write down the classification result and the squared error
    double error = square_error();
  
    if (label == predict) {
      ++nCorrect;
    }
  }

  // Summary
  double accuracy = (double)(nCorrect) / nTesting * 100.0;
  cout << "Node " << node << " test: \n";
  cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
  printf("Accuracy: %0.2lf\n", accuracy);
  
  report_test << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
  report_test << "Accuracy: " << accuracy << "\n" << endl;
}

int main (int argc, char *argv[])
{
  createTrainTestCSVs(60000, 10000, "4 - OpenMPI Multilayer Perceptron");
  int i, tag=1, tasks, iam;
  int start = 0, end = 0;
  MPI_Status status;

  struct timeval tval_before, tval_after, tval_result;
  gettimeofday(&tval_before, NULL);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    char hostname[HOST_NAME_MAX];
    char username[LOGIN_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    getlogin_r(username, LOGIN_NAME_MAX);

    read_dataset();

    init_arrays();

    if(iam == 0){
      fill_weights();

      copy_array(original_w1, final_w1, n1 * n2);
      copy_array(original_w2, final_w2, n2 * n3);
    } else {
      start = (iam-1) * batch_size;
      end = ((iam-1) * batch_size) + batch_size - 1;
    }

    MPI_Bcast(original_w1, n1*n2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(original_w2, n2*n3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(iam != 0){
      copy_array(original_w1, w1, n1 * n2);
      copy_array(original_w2, w2, n2 * n3);

      for(int epoch = 1; epoch <= epochs; epoch++){
        for(int sample = start; sample <= end; sample++){
          // Getting (image, label)
          input(in_dat[sample], out_dat[sample]);

          // Learning process: Perceptron (Forward procedure) - Back propagation
          int nIterations = learning_process();
        }
      }
      
      copy_array(w1, final_w1, n1 * n2);
      copy_array(w2, final_w2, n2 * n3);
      get_weights_difference(original_w1, w1, difference_w1, n1 * n2, nHosts - 1);
      get_weights_difference(original_w2, w2, difference_w2, n2 * n3, nHosts - 1);

      MPI_Send(difference_w1, n1 * n2, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
      MPI_Send(difference_w2, n2 * n3, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    } else {
      for(int i = 1; i <= nHosts - 1; i++){
        MPI_Recv(difference_w1, n1 * n2, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(difference_w2, n2 * n3, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);

        add_cumulative_array(difference_w1, final_w1, n1*n2);
        add_cumulative_array(difference_w2, final_w2, n2*n3);
      }

      gettimeofday(&tval_after, NULL);
      timersub(&tval_after, &tval_before, &tval_result);

      cout << "Time elapsed testing: " << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << " seconds" << endl;
      cout << endl;

      test_model(iam);
      write_matrix(model_fn);
    }

  MPI_Finalize();
  
  return 0;
}
