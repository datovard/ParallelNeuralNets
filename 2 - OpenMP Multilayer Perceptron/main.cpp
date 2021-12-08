// main.cpp
 
// don't forget to include out neural network
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include "NeuralNetwork.hpp"

using namespace std;

// Weights file name
const string model_fn = "output/model-neural-network";

// Report file name
const string report_fn = "output/training-report.dat";

// Number of training samples
int nTraining = 60000;

// File stream to write down a report
ofstream report;

int ReadCSV(std::string filename, bool** in_dat)
{
    std::ifstream file(filename);
    std::string line, word;

    // determine number of columns in file
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<bool> parsed_vec;

    while (getline(ss, word, ',')) {
        parsed_vec.push_back( (bool) (std::stoi(&word[0]))) ;
    }
    uint cols = parsed_vec.size();
    in_dat[0] = new bool[cols];
    for(int i = 0; i < cols; i++){
        in_dat[0][i] = parsed_vec[i];
    }
 
    // read the file
    int i = 1;
    if (file.is_open()) {
        while (getline(file, line, '\n') && i < nTraining) {
            std::stringstream ss(line);

            in_dat[i] = new bool[cols];
            uint j = 0;
            while (getline(ss, word, ',')) {
                in_dat[i][j] = (bool) std::stoi(&word[0]);
                j++;
            }
            i++;
        }
    }
    return cols;
}

void createCSVs(){
    const string training_image_fn = "../data/train-images.idx3-ubyte";
    const string training_label_fn = "../data/train-labels.idx1-ubyte";
    const string training_inputs_out = "input/training_input.csv";
    const string training_labels_out = "input/training_labels.csv";

    const string testing_image_fn = "../data/t10k-images.idx3-ubyte";
    const string testing_label_fn = "../data/t10k-labels.idx1-ubyte";
    const string testing_inputs_out = "input/testing_input.csv";
    const string testing_labels_out = "input/testing_labels.csv";

    ifstream training_image, testing_image;
    ifstream training_label, testing_label;
    ofstream training_inputs, testing_inputs;
    ofstream training_labels, testing_labels;

    training_inputs.open(training_inputs_out.c_str(), ios::out);
    training_labels.open(training_labels_out.c_str(), ios::out);
    training_image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    training_label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

    testing_inputs.open(testing_inputs_out.c_str(), ios::out);
    testing_labels.open(testing_labels_out.c_str(), ios::out);
    testing_image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    testing_label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

    const int width = 28;
    const int height = 28;
    const int n3 = 10;

    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        training_image.read(&number, sizeof(char));
        testing_image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        training_label.read(&number, sizeof(char));
        testing_label.read(&number, sizeof(char));
    }

    for (int sample = 1; sample <= nTraining; ++sample) {
        // Reading training images
        for (int j = 1; j <= height; ++j) {
            for (int i = 1; i <= width; ++i) {
                training_image.read(&number, sizeof(char));
                int pos = i + (j - 1) * width;
                training_inputs << ((number == 0)? 0: 1);
                if( pos < height * width )
                    training_inputs << ",";
            }
        }
        training_inputs << "\n";

        // Reading training labels
        training_label.read(&number, sizeof(char));
        for (int i = 1; i <= n3; ++i) {
            training_labels << (( i == number + 1 )? 1: 0);
            if( i < n3 )
            training_labels << ",";
        }
        training_labels << "\n";
    }

    for (int sample = 1; sample <= 10000; ++sample) {
        // Reading testing images
        for (int j = 1; j <= height; ++j) {
            for (int i = 1; i <= width; ++i) {
                testing_image.read(&number, sizeof(char));
                int pos = i + (j - 1) * width;
                testing_inputs << ((number == 0)? 0: 1);
                if( pos < height * width )
                    testing_inputs << ",";
            }
        }
        testing_inputs << "\n";

        // Reading testing labels
        testing_label.read(&number, sizeof(char));
        for (int i = 1; i <= n3; ++i) {
            testing_labels << (( i == number + 1 )? 1: 0);
            if( i < n3 )
            testing_labels << ",";
        }
        testing_labels << "\n";
    }

    training_image.close(); testing_image.close();
    training_label.close(); testing_label.close();
    training_inputs.close(); testing_inputs.close();
    training_labels.close(); testing_labels.close();
}

int main(int argc, char *argv[])
{
    createCSVs();
    struct timeval tval_before, tval_after, tval_result;

    cout << "Setting up...\n"; 
    report.open(report_fn.c_str(), ios::out);
    bool **in_dat = (bool **) malloc( (nTraining+1) * sizeof(bool*) );
    bool **out_dat = (bool **) malloc( (nTraining+1) * sizeof(bool*) );

    int input_cols = ReadCSV("input/training_input.csv", in_dat);
    int labels_cols =  ReadCSV("input/training_labels.csv", out_dat);
    
    int num_threads [] = {1, 2, 3, 4, 6, 8, 12, 24, 36}; 

    for(int i = 0; i < 9; i++){
        report << "Number of threads: " << num_threads[i] << "\n";
        for(int j = 0; j < 5; j++){
            gettimeofday(&tval_before, NULL);
            
            NeuralNetwork net(input_cols, labels_cols, num_threads[i]);

            int sample = 0;
            for( sample; sample < nTraining; sample++ ){
                net.updateInput(in_dat[sample]);
                net.updateExpectedOutput(out_dat[sample]);

                int nIterations = net.train();
            }

            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);

            net.write_matrix(model_fn + "_" + to_string(num_threads[i]) + "-" + to_string(j+1) + ".dat");
            report << "Sample " << sample << ", Error = " << net.calculateErrors();
            report << ", Time elapsed training: " << ((long int)tval_result.tv_sec) << "." << ((long int)tval_result.tv_usec) << " seconds" << endl ;
        }

        report << endl;
    }

    report.close();
    return 0;
}