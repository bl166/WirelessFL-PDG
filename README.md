# Wireless-PDGNet Implementation

This repo is related to the following paper:

Boning Li, Jake Perazzone, Ananthram Swami, and Santiago Segarra, "Learning to Transmit with Provable Guarantees in Wireless Federated Learning," submitted to IEEE TWC 2023. The preprint is available at https://arxiv.org/abs/2304.09329.

## Data Generation

Please refer to https://github.com/bmatthiesen/deep-EE-opt/tree/master/data to generate channel simulations. 

## Power Allocation

The implementation of the proposed model and utility functions can be found under `./PDGNet/`. 
For the training script, see `./main.py`. 

## Federated Learning

### 1. MNIST Digit Classification

For the iid experiment, please see `./FL-main-mnist.ipynb`.

For the non-iid experiment, please see `./FL-main-mnist-noniid.ipynb`.

### 2. UCI Air Quality Regression

Please see `./FL-main-text.ipynb`.

### 3. IMDB Sentiment Classification

Please see `./FL-main-text.ipynb`.

* _References are provided in the citation list and also as inline notes in code files._

