# Deep Learning Classification Project

<img width="100%" alt="ml-classification" src="https://github.com/MathieuMorel62/Machine_learning/assets/113856302/849e0a86-b7d2-4fec-a271-7347b4ee30dd">

## Description
This project aims to develop a deep neural network model for binary classification. Using the TensorFlow library, this project implements a classification model to predict two distinct classes from input data. Key features include network weight optimization, forward and backward propagation, and model performance evaluation.

## Data

The data used in this project is structured for binary classification. They include characteristics and associated labels for each entry. The data is pre-processed and formatted to be directly usable by the model. You can find the data in the [data](https://github.com/MathieuMorel62/Machine_learning/tree/main/supervised_learning/data) folder of the GitHub repository.

## Ressources
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)

## Prerequisites
- Python 3.5 or higher.
- TensorFlow 2.4 or higher.
- NumPy 1.15 or higher.
- Ubuntu 16.04 LTS

## Installation and Configuration

Make sure that all the prerequisites are installed. Then clone the repository from GitHub and install the necessary dependencies:

```bash
git clone https://github.com/MathieuMorel62/Machine_learning.git

cd supervised_learning/classification

./main.py

```

## Main Features
- **Forward and Back Propagation**: Implementation of algorithms for forward and backward propagation in the neural network.
- **Weight Optimization**: Use of the gradient descent algorithm to optimize network weights.
- **Model Evaluation**: Functions for evaluating the performance of the model on a set of test data.

<img width="382" alt="Capture d’écran 2024-01-07 à 18 56 09" src="https://github.com/MathieuMorel62/Machine_learning/assets/113856302/3a110b5d-5b9b-4b24-991a-11442aa5d083">


## List of Tasks
0. [**Neuron**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/0-neuron.py): Implement a class to represent a unique neuron in the network.
1. [**Privatize Neuron**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/1-neuron.py): Modify the Neuron class to make its attributes private and add getters/setters if necessary.
2. [**Neuron Forward Propagation**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/2-neuron.py): Develop the forward propagation method for a single neuron.
3. [**Neuron Cost**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/3-neuron.py): Create a function to calculate the cost (loss) of the neuron.
4. [**Evaluate Neuron**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/4-neuron.py): Write a method to evaluate neuron predictions.
5. [**Neuron Gradient Descent**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/5-neuron.py): Implement gradient descent for the neuron.
6. [**Neuron Train**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/6-neuron.py): Set up a training routine for the neuron.
7. [**Upgrade Train Neuron**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/7-neuron.py): Improve the neuron training routine.
8. [**NeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/8-neural_network.py): Create a class to represent a single-layer neural network.
9. [**Privatize NeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/9-neural_network.py): Make NeuralNetwork attributes private and add getters/setters.
10. [**NeuralNetwork Forward Propagation**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/10-neural_network.py): Develop the forward propagation method for the neural network.
11. [**NeuralNetwork Cost**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/11-neural_network.py): Create a function to calculate the cost of the neural network.
12. [**Evaluate NeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/12-neural_network.py): Write a method to evaluate network predictions.
13. [**NeuralNetwork Gradient Descent**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/13-neural_network.py): Implement gradient descent for the neural network.
14. [**Train NeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/14-neural_network.py): Set up a training routine for the neural network.
15. [**Upgrade Train NeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/15-neural_network.py): Improve the neural network training routine.
16. [**DeepNeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/16-deep_neural_network.py): Build a class for a multi-layered neural network.
17. [**Privatize DeepNeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/17-deep_neural_network.py): Make DeepNeuralNetwork attributes private and add getters/setters.
18. [**DeepNeuralNetwork Forward Propagation**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/18-deep_neural_network.py): Develop the forward propagation method for the deep neural network.
19. [**DeepNeuralNetwork Cost**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/19-deep_neural_network.py): Create a function to calculate the cost of the deep neural network.
20. [**Evaluate DeepNeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/20-deep_neural_network.py): Write a method to evaluate deep network predictions.
21. [**DeepNeuralNetwork Gradient Descent**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/21-deep_neural_network.py): Implement gradient descent for the deep network.
22. [**Train DeepNeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/22-deep_neural_network.py): Establish a training routine for the deep network.
23. [**Upgrade Train DeepNeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/23-deep_neural_network.py): Améliorer la routine d'entraînement du réseau profond.
24. [**One-Hot Encode**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/24-one_hot_encode.py): Implement one-hot encoding for labels.
25. [**One-Hot Decode**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/25-one_hot_decode.py): Develop a decoding for one-hot encoding.
26. [**Persistence is Key**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/26-deep_neural_network.py): Add features to save and load trained models.
27. [**Update DeepNeuralNetwork**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/27-deep_neural_network.py): Update the DeepNeuralNetwork class with new features or improvements.
28. [**All the Activations**](https://github.com/MathieuMorel62/Machine_learning/blob/main/supervised_learning/classification/28-deep_neural_network.py): Implement various activation functions in the network.

## Contact

- **LinkedIn Profile**: [Mathieu Morel](https://www.linkedin.com/in/mathieu-morel-9ab457261/)
- **GitHub Project Link**: [Classification Project](https://github.com/MathieuMorel62/Machine_learning/tree/main/supervised_learning/classification)
