#!/bin/bash

ENV_NAME="myenv"
conda activate $ENV_NAME
conda install -c conda-forge biopython
conda install numpy
conda install anaconda::pandas
conda install scikit-learn
conda install anaconda::scipy
conda install conda-forge::bio-embeddings-esm
pip install fair-esm
