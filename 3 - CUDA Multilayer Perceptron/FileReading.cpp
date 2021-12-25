#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

inline bool fileExists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

void createCSV(int numberSamples, std::string inputs, std::string labels, std::string outputFolder, std::string outputSuffix){
  const std::string testing_image_fn = inputs;
  const std::string testing_label_fn = labels;
  const std::string testing_inputs_out = "../" + outputFolder + outputSuffix + "_input.csv";
  const std::string testing_labels_out = "../" + outputFolder + outputSuffix + "_labels.csv";

  if( fileExists(testing_inputs_out) && fileExists(testing_labels_out) ){
    return;
  }

  std::ifstream testing_image, testing_label;
  std::ofstream testing_inputs, testing_labels;

  testing_inputs.open(testing_inputs_out.c_str(), std::ios::out);
  testing_labels.open(testing_labels_out.c_str(), std::ios::out);
  testing_image.open(testing_image_fn.c_str(), std::ios::in | std::ios::binary); // Binary image file
  testing_label.open(testing_label_fn.c_str(), std::ios::in | std::ios::binary ); // Binary label file

  const int width = 28;
  const int height = 28;
  const int n3 = 10;

  // Reading file headers
  char number;
  for (int i = 1; i <= 16; ++i) {
      testing_image.read(&number, sizeof(char));
  }
  for (int i = 1; i <= 8; ++i) {
      testing_label.read(&number, sizeof(char));
  }

  for (int sample = 1; sample <= numberSamples; ++sample) {
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

    testing_image.close();
    testing_label.close();
    testing_inputs.close();
    testing_labels.close();
}


void createTrainTestCSVs(int trainingSamples, int testingSamples, std::string outputFolder){
    const std::string training_image_fn = "../data/train-images.idx3-ubyte";
    const std::string training_label_fn = "../data/train-labels.idx1-ubyte";

    const std::string testing_image_fn = "../data/t10k-images.idx3-ubyte";
    const std::string testing_label_fn = "../data/t10k-labels.idx1-ubyte";
    const std::string output_fn = outputFolder + "/input/";

    createCSV(trainingSamples, training_image_fn, training_label_fn, output_fn, "training");
    createCSV(testingSamples, testing_image_fn, testing_label_fn, output_fn, "testing");
}

int ReadCSV(std::string filename, double** in_dat, int nTraining)
{
    std::ifstream file(filename);
    std::string line, word;

    // determine number of columns in file
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<double> parsed_vec;

    while (getline(ss, word, ',')) {
        parsed_vec.push_back( (double) (std::stod(&word[0]))) ;
    }
    uint cols = parsed_vec.size();
    in_dat[0] = new double[cols];
    for(int i = 0; i < cols; i++){
        in_dat[0][i] = parsed_vec[i];
    }
 
    // read the file
    int i = 1;
    if (file.is_open()) {
        while (getline(file, line, '\n') && i < nTraining) {
            std::stringstream ss(line);

            in_dat[i] = new double[cols];
            uint j = 0;
            while (getline(ss, word, ',')) {
                in_dat[i][j] = (double) std::stod(&word[0]);
                j++;
            }
            i++;
        }
    }
    return cols;
}