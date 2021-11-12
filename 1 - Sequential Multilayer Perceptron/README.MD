# Sequential Multilayer Perceptron

This folder contains a simple Multilayer Perceptron with sequential forward feeding and sequential backpropagation

## Explore this code!

You can run and experiment with this project more easily with this [Colab project!](https://colab.research.google.com/drive/1eQrv5FIrR-qY_J-JZuVO6YLF5A-_5IRP?usp=sharing)

## How to run this project

1. To compile this project you must run this line on your terminal:

       g++ main.cpp NeuralNetwork.cpp -o main

2. Having it compiled, to run your project you must start it with

       ./main
  
3. If you want to establish the number of hidden layers on the MLP, you can do it as:

       ./main  --layers 4 5 5 5 5
  
    Where the layers flag says that you want to set the number of neurons on each hidden layer.
    The first value after the flag says how many layers you want between the entry layer and the output layer of your MLP. After that, you must put the same quantity of numbers where each one says how many neurons you want on that layer.

4. Normally, this MLP is running with random generated testing data following a model of a 3-variable function:
   
       2x + 10y + 5z
   
    If you don't want to run with test generated data, you must supply your training data on the `data/data-in` and the training output on the `data/data-out` and run it with the flag:

       ./main  --noGenData

5. To set the size of the generated dataset, you can pass the flag:
  
       ./main  --dataSize 1000
    
    Where the `1000` value can be changed for any size you want!

