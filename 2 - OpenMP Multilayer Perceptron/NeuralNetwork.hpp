// NeuralNetwork.hpp
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
 
using namespace std;

// neural network implementation class!
class NeuralNetwork {
public:
    int width;
    int height;
    int n1;
    int n2 = 128; 
    int n3;
    const int epochs = 512;
    const double learning_rate = 1e-3;
    const double momentum = 0.9;
    const double epsilon = 1e-3;

    // From layer 1 to layer 2. Or: Input layer - Hidden layer
    double **w1, **delta1, *out1;

    // From layer 2 to layer 3. Or; Hidden layer - Output layer
    double **w2, **delta2, *in2, *out2, *theta2;

    // Layer 3 - Output layer
    double *in3, *out3, *theta3;
    double *expected;
    
    // constructor
    NeuralNetwork(int input_size, int label_size, int num_threads);
 
    // function for forward propagation of data
    void propagateForward();
 
    // function for backward propagation of errors made by neurons
    void propagateBackward();
 
    // function to calculate errors made by neurons in each layer
    double calculateErrors();
 
    // function to train the neural network give an array of data points
    int train();

    void write_matrix(string file_name);

    void updateInput(bool* referenceInput);
    
    void updateExpectedOutput(bool* referenceOutput);
};

