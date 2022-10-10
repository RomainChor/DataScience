########################################################################################
#                                                                                      #
#                            Code for NeurIPS 2022 submission ..                       #
#  Rate-Distortion Theoretic Bounds on Generalization Error for Distributed Learning   #
#                                                                                      #
########################################################################################

This archieve contains code and instructions for experiments for the paper:
Rate-Distortion Theoretic Bounds on Generalization Error for Distributed Learning


#################
# Requirements  #
#################
The code is tested with Python 3.6.7 on Linux. The following libraries
are required:

idx2numpy==1.2.3
matplotlib==3.0.3
numpy==1.16.3
pandas==0.24.2
scikit_learn==1.1.1
tqdm==4.64.0

Also, Jupyter Notebook (https://jupyter.org/install) is required to run experiments.

Jupyter : 4.4.0
Jupyter-notebook : 5.7.8


##################
# Code structure #
##################
 utils.py  # Contains all functions and classes necessary for the experiments
 run_experiments.ipynb	# Notebook to run experiments and get plots



#######################
# Running experiments #
#######################  
 Download MNIST data from: http://yann.lecun.com/exdb/mnist/.
 Extract the files and place in the current directory (or make sure to modify PATH variable in the notebook).
 Open run_experiments.ipynb notebook.
 Some parameters/hyperparameters can be changed, especially n, M and num_values which will influence the notebook's running time.
 Run all cells. Plots will appear in the last two cells. Uncomment last cell to save plots in .png files.

 Note: with default settings, the notebook can take approximatively 5 minutes to run.
