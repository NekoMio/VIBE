#!/usr/bin/env bash

export CONDA_ENV_NAME=vibe-env-m1
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7 pytorch

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r requirements.txt
