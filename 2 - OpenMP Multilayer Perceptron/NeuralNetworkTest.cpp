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
const string model_fn = "output/model-neural-network_12-1.dat";

// Report file name
const string report_fn = "output/testing-report_12-1.dat";

// Number of training samples
int nTesting = 10000;

// File stream to write down a report
ofstream report;

int predictionLabel( bool* labels ){
    int prediction = 0;
    for(int i = 0; i < 10; i++){
        if( labels[i] == 1 )
            prediction = i;
    }
    return prediction;
}

int main(int argc, char *argv[])
{
    struct timeval tval_before, tval_after, tval_result;

    cout << "Setting up...\n"; 
    report.open(report_fn.c_str(), ios::out);
    bool **in_dat = (bool **) malloc( (nTesting+1) * sizeof(bool*) );
    bool **out_dat = (bool **) malloc( (nTesting+1) * sizeof(bool*) );

    int input_cols = ReadCSV("input/testing_input.csv", in_dat, nTesting);
    int labels_cols =  ReadCSV("input/testing_labels.csv", out_dat, nTesting);
    
    gettimeofday(&tval_before, NULL);
    
    NeuralNetwork net(input_cols, labels_cols, 1);
    net.loadModel(model_fn);

    int nCorrect = 0;
    for( int sample = 0; sample < nTesting; sample++ ){
        net.updateInput(in_dat[sample]);
        net.updateExpectedOutput(out_dat[sample]);
        int label = predictionLabel(out_dat[sample]);

        net.propagateForward();

        double* output = net.getOutput();

        // Prediction
        int predict = 1;
        for (int i = 2; i <= labels_cols; ++i) {
			if (output[i] > output[predict]) {
				predict = i;
			}
		}
		--predict;

        if (label == predict) {
			++nCorrect;
			report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << net.calculateErrors() << endl;
		} else {
			report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << net.calculateErrors() << endl;
		}

        //report << "Values: Label = " << label << ", LABELS = " << *out_dat[0] << *out_dat[1] << *out_dat[2] << *out_dat[3] << *out_dat[4] << *out_dat[5] << *out_dat[6] << *out_dat[7] << *out_dat[8] << *out_dat[9] << endl;
        report << "LABELS = " << output[1]<< " "  << output[2]<< " "  << output[3]<< " "  << output[4]<< " "  << output[5]<< " "  << output[6]<< " "  << output[7]<< " "  << output[8]<< " "  << output[9] << " "  << output[10] << endl;
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;
    
    report << "Time elapsed testing: " << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << " seconds" << endl ;
    report << endl;
    
    report.close();
    return 0;
}