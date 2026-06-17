# Neurons that Learn on their Own: Neuron Autoencoders for MNIST and Spiking Neural Networks

This repository contains the code and experiments for the School of Engineering (SoE) Summer Internship at the **University of Manchester**, conducted under the supervision of **Dr. Davide Bertozzi**. 

The project is titled **"Neurons that Learn on their Own"** and focuses on implementing, scaling, and extending the neural architecture proposed by Dr. Larry Bull in his paper *"Neurons as Autoencoders"* (2025). Specifically, the work spans from reproducing the paper's original hill-climbing algorithm on tuneable NK fitness models, to scaling the approach with gradient descent in PyTorch on MNIST, and finally integrating it into Spiking Neural Networks (SNNs).

---

## Project Overview

In traditional Artificial Neural Networks, learning is driven globally by backpropagation. Dr. Larry Bull's paper proposes a biological-inspired alternative: **Neuron Autoencoders**, where each hidden neuron acts as its own local autoencoder. The neuron sends a signal forward (axon) but also attempts to reconstruct its inputs locally (dendritic backward echo).

This project explores:
1. **Faithful Reproduction**: Implementing the random hill-climber training algorithm on NK regression landscapes exactly as described in the paper.
2. **Gradient-Based Translation**: Swapping the hill-climber for gradient descent in PyTorch to scale the architecture to complex data.
3. **MNIST Scaling**: Transitioning the neuron-autoencoder from synthetic toy regression to image classification on MNIST.
4. **Spiking Neural Networks (SNNs)**: Extending the neuron-autoencoder to temporal spiking dynamics using Leaky Integrate-and-Fire (LIF) neurons and surrogate gradients.
5. **Training Schedules**: Testing interleaved training (alternating task and reconstruction cycles) versus sequential training (pre-training autoencoders, then freezing and training the task head).
6. **Neuron Collapse Diagnostics & Prevention**: Investigating whether hidden neurons collapse onto the same representations by measuring activation correlation and effective rank, and introducing diversity regularizers to encourage representation variety.

---

## Repository Structure

The project is structured sequentially across several stages:

### 1. NK Model & Faithful Hill-Climbing
* File: [`stage1_2_bull_faithful.py`](stage1_2_bull_faithful.py)
* **Stage 1**: Implements Kauffman's NK model as a tuneable regression data source.
* **Stage 2**: Implements a NumPy-only reproduction of the paper's three architectures:
  * **NN**: Plain MLP (baseline).
  * **ANN**: One shared decoder layer on the hidden layer.
  * **NAN**: Each hidden neuron has its own individual decoder (Neuron Autoencoder Network).
  * Trained strictly using the paper's random hill-climber (no backpropagation).

### 2. PyTorch Gradients & Spiking Networks
* File: [`stage3_4_gradient_and_spiking.py`](stage3_4_gradient_and_spiking.py)
* **Stage 3**: Adapts the NAN architecture to PyTorch so that autograd can train the encoder/decoder weights.
* **Stage 4**: Extends the local autoencoding bottleneck to Spiking Neural Networks. Each neuron's activation code is defined by its average firing rate over $T$ timesteps, optimized using surrogate gradients.

### 3. MNIST Porting
* File: [`stage5_mnist.py`](stage5_mnist.py)
* **Stage 5**: Adapts both the gradient and spiking NAN networks to MNIST classification. Translates the NK difficulty knobs ($N$ and $K$) to image resolution (downscaling) and class density/noise.

### 4. Training Schedules & Diagnostics
* File: [`stage6_schedules_and_collapse.py`](stage6_schedules_and_collapse.py)
* **Stage 6**: Explores the training schedule and the "neuron collapse" question:
  * Adds interleaved vs. sequential training modes.
  * Implements diagnostics to measure representation health: decoder cosine similarity, activation correlation, and the effective rank of the activation matrix.
  * Introduces a decoder diversity penalty to actively prevent collapse.

### 5. Multi-Seed Experiment Harness
* File: [`stage7_experimental_sweeps.py`](stage7_experimental_sweeps.py)
* **Stage 7**: An instrumented test harness that aggregates training metrics across multiple seeds. It sweeps hyperparameters (such as reconstruction weights and diversity penalties) and outputs curves and performance logs to the `/experiments` directory.

### 6. Interactive Documentation
* File: [`autoencoders_internship.ipynb`](autoencoders_internship.ipynb)
* An interactive Jupyter Notebook walking through explanations, code runs, and visualizations of the experiments.

---

## Requirements

To run the experiments, make sure you have the following packages installed:
* `numpy`
* `torch`
* `torchvision`
* `matplotlib`

---

## Getting Started

To run a quick demonstration of the multi-seed experiment harness:
```bash
python stage7_experimental_sweeps.py
```
This runs a demo comparing static and spiking neuron autoencoders, saving the results in a local `experiments/` directory.