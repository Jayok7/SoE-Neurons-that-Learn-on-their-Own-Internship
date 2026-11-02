# Neurons that Learn on their Own: Neuron Autoencoders for MNIST and Spiking Neural Networks

This repository contains the code and experiments for a Summer Internship.

The project is titled **"Neurons that Learn on their Own"** and focuses on implementing, scaling, and extending the neural architecture proposed by Dr. Larry Bull in his paper *"Neurons as Autoencoders"* (2025). Specifically, the work spans from reproducing the paper's original hill-climbing algorithm on tuneable NK fitness models, to scaling the approach with gradient descent in PyTorch on MNIST, and finally integrating it into Spiking Neural Networks (SNNs).

---

## Project Overview

In traditional Artificial Neural Networks, learning is driven globally by backpropagation. Dr. Larry Bull's paper proposes a biological-inspired alternative: **Neuron Autoencoders**, where each hidden neuron acts as its own local autoencoder. The neuron sends a signal forward (axon) but also attempts to reconstruct its inputs locally (dendritic backward echo).

This project explores:

1. **Faithful Reproduction**: Implementing the random hill-climber training algorithm on NK regression landscapes based on the descriptions on the paper.
2. **Gradient-Based Translation**: Swapping the hill-climber for gradient descent in PyTorch to scale the architecture to complex data.
3. **MNIST Scaling**: Transitioning the neuron-autoencoder from synthetic toy regression to image classification on MNIST.
4. **Spiking Neural Networks (SNNs)**: Extending the neuron-autoencoder to temporal spiking dynamics using Leaky Integrate-and-Fire (LIF) neurons and surrogate gradients.
5. **Training Schedules**: Testing interleaved training (alternating task and reconstruction cycles) versus sequential training (pre-training autoencoders, then freezing and training the task head) vs hybrid (pre-training autoencoders, then running interleaved training cycles).
6. **Neuron Collapse Diagnostics & Prevention**: Investigating whether hidden neurons collapse onto the same representations by measuring activation correlation and effective rank, and introducing diversity regularizers to encourage representation variety.

---

## Requirements

To run the experiments, make sure you have the following packages installed:

* `numpy`
* `torch`
* `torchvision`
* `matplotlib`

---
