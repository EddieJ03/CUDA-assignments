# CSE 160 Programming Assignment 8: (Optimized) CNN Forward Layer GPU Implementation 

## Objective

This is the last part of a three part project implementing and optimizing the forward pass of a convolution layer using CUDA. Convolutional layers are the primary building blocks of convolutional neural networks (CNNs), which are used for tasks like image classification, object detection, natural language processing and recommendation systems. 

You will be working with a modified version of the LeNet5 architecture shown below:

![LenetImage](https://lh5.googleusercontent.com/84RlneM7JSDYDirUr_ceplL4G3-Peyq5dkLJTe2f-3Bj9KuWZjsH2A9Qq5PO5BRLrVfWGPnI3eQu8RkTPgyeUf9ZOWY9JbptVJy9LceAyHRn-O0kbzprx88yb82a5dnCR7EDP7n0)

You can read about the original network:

*Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf*

Your optimized CUDA implementation of the convolutional layer will be used to perform inference for layers C1 and C3 (shown in red) in the figure above. This leverages the [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) (Mini-DNN) framework for implementing the modified LeNet-5.

## Input Data

The network will be tested on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) which contains 10,000 single channel images each of dimensions 86x86 but we will only use 1000 of these at a time. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot, etc).

## Instructions

This assignment requires you to complete a GPU implementation of the convolutional layer. Performance of the GPU implementation is not important as this assignment is intended to build functionality before optimizing. The only file you need to update to implement the forward convolution is:
`src/layer/custom/new-forward.cu`. 
