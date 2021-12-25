%%writefile testing.cu

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
#include <sys/time.h>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// Training image file name
const string training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "mnist/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network-mix-10k.dat";

// Report file name
const string report_fn = "training-report-mix-10k.dat";

// Number of training samples
const int nTraining = 1;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n3 = Number of neurons per layer
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

int n1;
int n2;
int n3;
const int epochs = 1;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1, *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2;

// Layer 3 - Output layer
double *out3;
double *expected;

double *w1_d, *w2_d;
double *in2_d, *in3_d;
double *out1_d, *out2_d, *out3_d;
double *delta1_d, *delta2_d;
double *theta2_d, *theta3_d;
double *expected_d;

double *in_dat_d, *out_dat_d;

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

__global__ void
multiplyWithTransposedMatrix(const double *weights, const double *outputs, double *inputs, int height, int width, int sample)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  printf("ENTRA KERNEL outputs: %f\n", outputs[0]);
  int i;
  
  if (j < width)
  {
    for (i = 0; i < height; i++)
    {
      sum += weights[i * width + j] * outputs[i + (sample * height)];
    }
    printf("\n");
  }
  __syncthreads();
  inputs[j] = sum;
}

__global__ void
applySigmoid(const double *inputs, double *outputs, const int size)
{
  __shared__ double in_buffer[1024];
  __shared__ double out_buffer[1024];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  in_buffer[i] = inputs[i];
  __syncthreads();

  if (i < size)
  {
    out_buffer[i] = 1.0 / (1.0 + exp(-1 * in_buffer[i]));
    __syncthreads();
    outputs[i] = out_buffer[i];
  }
}

__global__ void
applyLastTheta(const double *outputs, const double *expecteds, double *theta, int size, int sample)
{
  __shared__ double out_buffer[1024];
  __shared__ double exp_buffer[1024];
  __shared__ double tht_buffer[1024];
  int i = threadIdx.x;
  out_buffer[i] = outputs[i];
  exp_buffer[i] = expecteds[i + (sample * size)];
  __syncthreads();

  if (i < size)
  {
    tht_buffer[i] = out_buffer[i] * (1 - out_buffer[i]) * (((exp_buffer[i] == 0) ? 0 : 1) - out_buffer[i]);
    __syncthreads();
    theta[i] = tht_buffer[i];
  }
}

__global__ void
applyInnerTheta(const double *weights, const double *lastTheta, const double *outputs, double *innerTheta, int height, int width)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0, total;
  __shared__ double out_buffer[1024];
  out_buffer[i] = outputs[i];
  __syncthreads();

  if (i < height)
  {
    for (int j = 0; j < width; j++)
    {
      sum += weights[i * width + j] * lastTheta[j];
    }
    total = out_buffer[i] * (1 - out_buffer[i]) * sum;
    __syncthreads();

    innerTheta[i] = total;
  }
}

__global__ void
applyLayerPropagation(const double *theta, const double *outputs, double *delta, double *weights, int height, int width, double learning_rate, double momentum)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double delta_local;
  __shared__ double out_buffer[1024];
  __shared__ double tht_buffer[1024];
  out_buffer[i] = outputs[i];
  if (i < width)
  {
    tht_buffer[i] = theta[i];
  }
  __syncthreads();

  if (i < height)
  {
    for (int j = 0; j < width; j++)
    {
      delta_local = (learning_rate * tht_buffer[j] * out_buffer[i]) + (momentum * delta[i * width + j]);
      delta[i * width + j] = delta_local;
      weights[i * width + j] += delta_local;
    }
  }
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array()
{
  n1 = width * height;
  n2 = 128;
  n3 = 10;

  w1 = new double[n1 * n2];
  w2 = new double[n2 * n3];
  out1 = new double[n1];
  out3 = new double[n3];
  expected = new double[n3];

  // Initialization for weights from Input layer to Hidden layer
  for (int i = 0; i < n1 * n2; i++)
  {
    w1[i] = ((rand() % 2 == 1) ? -1 : 1) * (double)(rand() % 6) / 10.0;
  }

  // Initialization for weights from Hidden layer to Output layer
  for (int i = 0; i < n2 * n3; i++)
  {
    w2[i] = ((rand() % 2 == 1) ? -1 : 1) * (double)(rand() % 10 + 1) / (10.0 * n3);
  }

  cudaMalloc(&w1_d, n1 * n2 * sizeof(double));
  cudaMalloc(&w2_d, n2 * n3 * sizeof(double));

  cudaMalloc(&in2_d, n2 * sizeof(double));
  cudaMalloc(&in3_d, n3 * sizeof(double));

  cudaMalloc(&out1_d, n1 * sizeof(double));
  cudaMalloc(&out2_d, n2 * sizeof(double));
  cudaMalloc(&out3_d, n3 * sizeof(double));

  cudaMalloc(&delta1_d, n1 * n2 * sizeof(double));
  cudaMalloc(&delta2_d, n2 * n3 * sizeof(double));

  cudaMalloc(&theta2_d, n2 * sizeof(double));
  cudaMalloc(&theta3_d, n3 * sizeof(double));

  cudaMalloc(&expected_d, n3 * sizeof(double));

  cudaMemcpy(w1_d, w1, n1 * n2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(w2_d, w2, n2 * n3 * sizeof(double), cudaMemcpyHostToDevice);
}

void free_array()
{
  cudaFree(w1_d);
  cudaFree(w2_d);

  cudaFree(in2_d);
  cudaFree(in3_d);

  cudaFree(out1_d);
  cudaFree(out2_d);
  cudaFree(out3_d);

  cudaFree(delta1_d);
  cudaFree(delta2_d);

  cudaFree(theta2_d);
  cudaFree(theta3_d);

  cudaFree(expected_d);
}

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+

void perceptron(int sample)
{
  cudaMemset(in2_d, 0, n2 * sizeof(double));
  cudaMemset(in3_d, 0, n3 * sizeof(double));

  multiplyWithTransposedMatrix<<<1, n2>>>(w1_d, in_dat_d, in2_d, n1, n2, sample);

  applySigmoid<<<1, n2>>>(in2_d, out2_d, n2);

  multiplyWithTransposedMatrix<<<1, n3>>>(w2_d, out2_d, in3_d, n2, n3, 0);

  applySigmoid<<<1, n3>>>(in3_d, out3_d, n3);

  cudaMemcpy(out3, out3_d, n3 * sizeof(double), cudaMemcpyDeviceToHost);
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error()
{
  double res = 0.0;
  for (int i = 0; i < n3; i++)
  {
    res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
  }
  res *= 0.5;
  return res;
}

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+

void back_propagation(int sample)
{
  applyLastTheta<<<1, n3>>>(out3_d, out_dat_d, theta3_d, n3, sample);

  applyInnerTheta<<<1, n2>>>(w2_d, theta3_d, out2_d, theta2_d, n2, n3);

  applyLayerPropagation<<<1, n2>>>(theta3_d, out2_d, delta2_d, w2_d, n2, n3, learning_rate, momentum);

  applyLayerPropagation<<<1, n1>>>(theta2_d, out1_d, delta1_d, w1_d, n1, n2, learning_rate, momentum);
}

// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+

int learning_process(int sample)
{
  cudaMemset(delta1_d, 0, n1 * n2 * sizeof(double));
  cudaMemset(delta2_d, 0, n2 * n3 * sizeof(double));

  for (int i = 0; i < epochs; i++)
  {
    perceptron(sample);
    back_propagation(sample);
    if (square_error() < epsilon)
    {
      return i;
    }
  }
  return epochs;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

void input()
{
  // Reading image
  char number;
  for (int j = 0; j < height; j++)
  {
    for (int i = 0; i < width; i++)
    {
      image.read(&number, sizeof(char));

      int pos = i + j * width;
      out1[pos] = (number == 0) ? 0 : 1;
    }
  }

  cudaMemcpy(out1_d, out1, n1 * sizeof(double), cudaMemcpyHostToDevice);

  // Reading label
  label.read(&number, sizeof(char));
  for (int i = 0; i < n3; ++i)
  {
    expected[i] = 0.0;
  }
  expected[number] = 1.0;

  cudaMemcpy(expected_d, expected, n3 * sizeof(double), cudaMemcpyHostToDevice);
}

// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name)
{
  cudaMemcpy(w2, w2_d, n2 * n3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(w1, w1_d, n1 * n2 * sizeof(double), cudaMemcpyDeviceToHost);

  ofstream file(file_name.c_str(), ios::out);

  // Input layer - Hidden layer
  for (int pos = 0; pos < n1 * n2; pos++)
  {
    file << w1[pos] << " ";

    if (pos % width == n2)
      file << endl;
  }

  // Hidden layer - Output layer
  for (int pos = 0; pos < n2 * n3; pos++)
  {
    file << w2[pos] << " ";

    if (pos % width == n3)
      file << endl;
  }

  file.close();
}

// +--------------+
// | Main Program |
// +--------------+

thrust::host_vector<double> ReadCSV(std::string filename, int nTraining)
{
    thrust::host_vector<double> data;
    std::ifstream file(filename);
    std::string line, word;
 
    // read the file
    int i = 1;
    if (file.is_open()) {
        while (getline(file, line, '\n') && i < nTraining) {
            std::stringstream ss(line);

            while (getline(ss, word, ',')) {
                data.push_back((double) std::stod(&word[0]));
            }
            i++;
        }
    }
    return data;
}

int main(int argc, char *argv[])
{
  struct timeval tval_before, tval_after, tval_result;

  report.open(report_fn.c_str(), ios::out);

  thrust::host_vector<double> in_dat   = ReadCSV("input/training_input.csv", nTraining);
  thrust::host_vector<double> out_dat  = ReadCSV("input/training_labels.csv", nTraining);

  thrust::device_vector<double> in_step = in_dat;
  thrust::device_vector<double> out_step = out_dat;

  /*double *in_step = (double *) in_dat.data();
  double *out_step = (double *) out_dat.data();*/

  in_dat_d = thrust::raw_pointer_cast(in_step.data());
  out_dat_d = thrust::raw_pointer_cast(out_step.data());

  /*cudaMalloc(&in_dat_d, n1 * nTraining * sizeof(double));
  cudaMalloc(&out_dat_d, n3 * nTraining * sizeof(double));

  cudaMemcpy(in_dat_d, in_step, n1 * nTraining * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(out_dat_d, out_step, n3 * nTraining, out_dat.size() * sizeof(double), cudaMemcpyHostToDevice);*/

  /*for( int j = 0; j < 3; j++ ){
    for( int i = 0; i < 28*28; i++ ){
      if(i % 28 == 0) cout << "\n";
      cout << in_dat[(j * 28*28) + i] << " ";
    }
    cout << "\n";
  }

  for( int j = 0; j < 3; j++ ){
    for( int i = 0; i < 10; i++ ){
      cout << in_dat[(j * 10) + i] << " ";
    }
    cout << "\n";
  }*/

  // Neural Network Initialization
  init_array();

  gettimeofday(&tval_before, NULL);
  report << "sample,iterations,square error" << endl;
  cout << "Training started" << endl;
  int counter = 0;
  for (int sample = 1; sample <= nTraining; ++sample)
  {
    // Getting (image, label)
    //input();

    // Learning process: Perceptron (Forward procedure) - Back propagation
    int nIterations = learning_process(sample - 1);
    if (nIterations == epochs)
      counter++;

    if (sample % 500 == 0)
    {
      cout << sample << ": " << nIterations << ": " << counter << endl;
      if (counter == sample)
      {
        cout << "All the same" << endl;
        break;
      }

      // Write down the squared error
      gettimeofday(&tval_after, NULL);
      timersub(&tval_after, &tval_before, &tval_result);
      report << sample << "," << nIterations << "," << square_error() << "," << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << endl;
    }

    if (sample % 1000 == 0)
    {
      write_matrix(model_fn);
    }
  }

  gettimeofday(&tval_after, NULL);
  timersub(&tval_after, &tval_before, &tval_result);

  report << "Time elapsed testing: " << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << " seconds" << endl;
  report << endl;

  // Save the final network
  write_matrix(model_fn);

  free_array();

  image.close();
  label.close();
  report.close();

  return 0;
}
