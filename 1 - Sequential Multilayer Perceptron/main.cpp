// main.cpp
 
// don't forget to include out neural network
#include <fstream>
#include <iostream>
#include <string>
#include "NeuralNetwork.hpp"
 
void genData(std::string filename)
{
    std::ofstream file1(filename + "-in");
    std::ofstream file2(filename + "-out");
    for (uint r = 0; r < 100000; r++) {
        Scalar x = rand() / Scalar(RAND_MAX);
        Scalar y = rand() / Scalar(RAND_MAX);
        Scalar z = rand() / Scalar(RAND_MAX);
        file1 << x << ", " << y << ", " << z << std::endl;
        file2 << (2 * x + 10 * y + 5 * z) << std::endl;
    }
    file1.close();
    file2.close();
}

void ReadCSV(std::string filename, std::vector<RowVector*>& data)
{
    data.clear();
    std::ifstream file(filename);
    std::string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<Scalar> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.push_back(Scalar(std::stof(&word[0])));
    }
    uint cols = parsed_vec.size();
    data.push_back(new RowVector(cols));
    for (uint i = 0; i < cols; i++) {
        data.back()->coeffRef(1, i) = parsed_vec[i];
    }
 
    // read the file
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(new RowVector(1, cols));
            uint i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                i++;
            }
        }
    }
}

typedef std::vector<RowVector*> data;
int main(int argc, char *argv[])
{
    std::vector<uint> hiddenLayers;
    data in_dat, out_dat;
    bool noGenData = false;

    if( argc > 1 ){
        int i = 1;
        while( i < argc ){
            std::string argument = argv[i];
            
            if( !argument.compare("--layers") ){
                int quantity = std::stoi(argv[i + 1]);
                i += 2;
               
                int limit = i + quantity - 1;
                for( i; i <= limit; i++ ){
                    hiddenLayers.push_back(std::stoi(argv[i]));
                }
            }else if( !argument.compare("--noGenData") ){
                noGenData = true;
                ++i;
            }else {
                std::cout << "Unknown parameter: " << argv[i] << "\n";
                return 0;
            }            
        }
    }

    if(!noGenData){
        genData("./data/data");
    }

    ReadCSV("./data/data-in", in_dat);
    ReadCSV("./data/data-out", out_dat);

    std::vector<uint> topology;
    topology.push_back(in_dat[0]->size());
    for( int i = 0; i < hiddenLayers.size(); i++ ){
        topology.push_back(hiddenLayers[i]);
    }
    topology.push_back(out_dat[0]->size());

    NeuralNetwork n(topology);
    n.train(in_dat, out_dat);
    return 0;
}