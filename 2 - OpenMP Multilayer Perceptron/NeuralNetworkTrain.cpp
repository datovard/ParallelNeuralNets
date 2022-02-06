// main.cpp
 
// don't forget to include out neural network
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <sstream>
#include "NeuralNetwork.hpp"
#include "../util/FileReading.cpp"

using namespace std;

// Weights file name
const string model_fn = "output/model-neural-network";

// Report file name
const string report_fn = "output/training-report.dat";

// Number of training samples
int nTraining =60000;

// File stream to write down a report
ofstream report;

int main(int argc, char *argv[])
{
    createTrainTestCSVs(60000, 10000, "2 - OpenMP Multilayer Perceptron");
    struct timeval tval_before, tval_after, tval_result;

    cout << "Setting up...\n"; 
    report.open(report_fn.c_str(), ios::out);
    bool **in_dat = (bool **) malloc( (nTraining+1) * sizeof(bool*) );
    bool **out_dat = (bool **) malloc( (nTraining+1) * sizeof(bool*) );

    int input_cols = ReadCSV("input/training_input.csv", in_dat, nTraining);
    int labels_cols =  ReadCSV("input/training_labels.csv", out_dat, nTraining);
    
    int num_threads [] = {12}; 

    cout << "Training...\n";
    for(int i = 0; i < 1; i++){
        report << "Number of threads: " << num_threads[i] << endl;
        for(int j = 0; j < 1; j++){
            gettimeofday(&tval_before, NULL);
            
            NeuralNetwork net(input_cols, labels_cols, num_threads[i]);

            int sample = 1, counter = 0;
            for( sample; sample <= nTraining; ++sample ){
                net.updateInput(in_dat[sample-1]);
                net.updateExpectedOutput(out_dat[sample-1]);

                int nIterations = net.train();
                if(nIterations == net.epochs) counter++;

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);

                if(sample % 500 == 0){
                    report << sample << "," << nIterations << "," << net.calculateErrors() << "," << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << endl;
                    cout << sample << ": " << nIterations << ": " << counter << endl;
                }
            }

            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);

            net.writeMatrix(model_fn + "-" + to_string(num_threads[i]) + "-" + to_string(j+1) + "-1.dat");
            report << "Sample " << sample << ", Error = " << net.calculateErrors();
            report << ", Time elapsed training: " << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << " seconds" << endl ;
        }

        report << endl;
    }

    report.close();
    return 0;
}