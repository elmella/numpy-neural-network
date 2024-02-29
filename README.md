# NumPy Neural Network vs. XGBoost Classification

This project showcases the implementation of a neural network from scratch using NumPy for digit classification on the MNIST dataset and compares its performance with the XGBoost classifier. The neural network demonstrates core deep learning concepts through practical application, including forward and backward propagation.

## Project Overview

The aim is to build a custom neural network to understand and implement the detailed mechanics of deep learning algorithms, particularly focusing on the mathematical underpinnings of model training and prediction processes.

## Core Concepts and Mathematical Foundation

### Forward Propagation

Forward propagation involves calculating the output of the neural network for a given input. This process sequentially computes and passes activations from one layer to the next in the network.

1. **Input to Hidden Layer Transformation**

   $$Z^{[1]} = W^{[1]}X + b^{[1]}$$

2. **Activation Function (ReLU)**

   $$A^{[1]} = ReLU(Z^{[1]}) = \max(0, Z^{[1]})$$

3. **Output Layer Transformation**

   $$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$

4. **Softmax Activation for Output Layer**

   $$A^{[l]} = \text{softmax}(Z^{[l]}) = \frac{e^{Z^{[l]}}}{\sum e^{Z^{[l]}}}$$

### Backward Propagation

Backward propagation computes the gradient of the loss function with respect to each weight and bias in the network, which are then used to update the parameters.

1. **Gradient of Loss with Respect to Output**

   $$\frac{\partial L}{\partial A^{[l]}} = A^{[l]} - Y$$

2. **Gradients of Weights and Biases**

   $$\frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} \frac{\partial L}{\partial A^{[l]}} A^{[l-1]T}$$

   $$\frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum \frac{\partial L}{\partial A^{[l]}}$$

3. **Gradient of Loss with Respect to Activation Function (ReLU)**

   The gradient of the loss with respect to the activation of the previous layer can be computed as:

   $$\frac{\partial L}{\partial A^{[l-1]}} = (W^{[l]T} \frac{\partial L}{\partial A^{[l]}}) * g'(Z^{[l-1]})$$

  where $g'(Z)$, the derivative of the ReLU function, is defined as:

   $$g'(Z) = 
   \begin{cases} 
   1 & \text{if } Z > 0, \\
   0 & \text{otherwise.}
   \end{cases}$$

### Implementation Highlights

- **NumPy for Mathematical Operations**: Demonstrates the use of NumPy to efficiently perform matrix and vector operations essential for neural network computations.
- **Customizable Neural Network Architecture**: Allows for experimentation with different layer sizes, activation functions, and optimization techniques.


### Conclusion

This project illustrates the fundamental concepts of building and training a neural network using NumPy, from initializing parameters to performing forward and backward propagation. Just to demonstrate that deep learning is not the end-all, be-all of machine learning, I train XGBoost on the same model and blow it out of the water
