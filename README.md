## Federated Learning Algorithms with Heterogeneous Data Distributions: An Empirical Evaluation

This repository contains TensorFlow codes to simulate different FL algorithms to limit the degradation introduced by 
non-IID data distributions. 

## Algorithms
List of implemented algorithms in this repo:
- FedProx
- FedGKD
- FedNTD
- FedMLB
- FedDyn
- MOON
- FedAvgM
- FedAdam
- FedAvgM

Default values.
- For FedProx we tuned $\mu$ in $\{0.01, 0.001\}$. $\mu$ controls the weight of the proximal term in the local objective function.
- For FedGKD we set $\gamma$ to 0.2, as in the original paper. $\gamma$ controls the weight of the  KD-based term in the local objective function.
- For FedNTD we selected $\lambda$ in $\{0.3, 1.0\}$.
- For FedMLB $\lambda_1$ and $\lambda_2$ are both set to 1 ($\lambda_1$, $\lambda_2$ weight the impact of the hybrid cross-entropy loss and the KD-based loss, see Fig. \ref{fig:fedmlb}). 5 blocks are considered, formed as in the original paper, where conv1, conv2\_x, conv3\_x, conv4\_x, conv5\_x and the fully connected layer constitutes a single block. 
- For FedAvgM we selected the momentum parameter among $\{ 0.4, 0.6, 0.8, 0.9\}$.
- For FedAdam we set $\tau$ (a constant for numerical stability) equal to 0.001 as in \cite{kim2022multi, reddi2020adaptive}.
- For FedDyn we set $\alpha$ equal to 0.1 as in the original paper.

# Dataset and Model Architecture 
[cifar100](https://www.tensorflow.org/datasets/catalog/cifar100) and ResNet-18.

### Data partitioning
The CIFAR100 dataset is partitioned following the paper [Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335): a Dirichlet distribution is used to decide the per-client label distribution. 
A concentration parameter controls the identicalness among clients. Very high values for such a concentration parameter, `alpha` in the code, (e.g., > 100.0) imply an identical distribution of labels among clients,
while low (e.g., 1.0) values produce a very different amount of examples for each label in clients, and for very low values (e.g., 0.1) all the client's examples belong to a single class.

### Instructions
`fed_resnet8.py` contains the simulations code for all the algorithms.
Hyperparameters can be choosen by manually modifying the `hp` dictionary. A simulation of each combination of hyperparameters will be run.

Similarly, `simulation_feddf.py` contains the simulations code for FedDF.
Note: before running the simulation(s) the partitioned cifar100 must be created using the provided script (see the following).

The client-side algorithms (FedProx, FedGKD, FedNTD, FedMLB, MOON, FedDyn) are implemented by subclassing the 
`tf.keras.Model` class, and overwriting the `train_step` and `test_step` methods.

#### Creating a virtual environment with venv
`python3 -m venv fd_env`

`source fd_env/bin/activate`

`pip install -r requirements.txt`

The code has been tested with `python==3.8.10`.

Note: to run FedDyn algorithm, `tf == 2.11.0` or above is needed, but it has to be changed manually. 
We did not include it in the requirements by default because that version has still memory leaks 
as I pointed out [here](https://github.com/keras-team/keras/issues/17458).

#### Creating a virtual environment with venv
Running the simulation(s).

`python3 simulation.py`

`python3 simulation_feddf.py`

#### Creating partitioned CIFAR100   
Before running `simulation.py`, the partitioned CIFAR10 dataset must be generated by executing `dirichlet_partition.py`. 
The script will create a `cifar10_alpha` folder inside the current directory, with three values for alpha (0.1, 1.0, 100.0). This directory will 
contain a folder for each `client` with their examples.

If possible, the `dirichlet_partition.py` will create disjoint dataset for clients.

#### Logging

The `simulation.py` produces logging txt files with per-round metrics. It also creates tensorboard logging files with global model accuracy.



