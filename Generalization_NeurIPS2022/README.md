#                            Code for NeurIPS 2022 paper                     #
#  Rate-Distortion Theoretic Bounds on Generalization Error for Distributed Learning   #
Authors: M. Sefidgaran, R. Chor, A. Zaidi  
Available at [arxiv.org/abs/2206.02604](https://arxiv.org/abs/2206.02604).

This archive contains code and instructions for reproducing experiments in the paper titled "Rate-Distortion Theoretic Bounds on Generalization Error for Distributed Learning" presented at the NeurIPS 2022 conference, which took place in New Orleans, Louisiana (USA) from Monday November 28th through Friday December 9th.  


# Requirements 
The code is tested with Python 3.6.7 on Linux. Please see requirements.txt.


# Code structure 
**utils/**: contains utility files.

**run_experiments.py**: main code for running experiments.
Arguments:  
- `--data_path`: Path to directory containing MNIST data, type: str  
- `--save_path`: Path to directory where to save experiments results and plots, type: str  
- `--name`: Dataset name, type: str  
- `--classes`: MNIST classes, type: sequence of 2 ints  
- `--K_values`: List of values of K (number of clients), type: sequence of ints  
- `--n`: Client dataset size, type: int  
- `--proj_dim`: Dimension of SVM kernel space, type: int  
- `--gamma`: SVM kernel scaling parameter, type: int  
- `--client_epochs`: Number of epochs for local training, type: int
- `--epochs`: Number of epochs for centralized training, type: int  
- `--lr`: Learning rate, type: float  
- `--MC`: Number of runs (Monte-Carlo simulations), type: int


# Running experiments 
Our implementation requires manually downloading MNIST data from e.g., [yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).  
Extract the files as per your convenience and make sure to modify the `--save_path` argument.  
To reproduce the simulations that allowed us to produce Fig. 2 and 3 in the paper, run the following command:  
`python run_experiments.py --data_path "" --save_path "save/comp_K" --name "mnist" --classes 1 6 --K_values 2 5 10 25 50 75 100 125 --n 100 ---proj_dim 2000 --gamma 0.01 --client_epochs 200 --epochs 200 --lr 0.01 --MC 100`  
where `--save_path` is to be set to the folder containing the MNIST data, and `--n` is to be set to 100 or 300. 
