# Description

VisINet: automating visual inspection for protein-ligand docking results

The source code is an image-based virtual screening method named VisINet (VISual Inspection Network), where compounds were screened solely based on images taken from protein-ligand docking pose and known activity of ligands, without any other explicit information of ligands, such as molecular structures or docking scores.

# Installation

- This model is tested on linux, with python 3.5, cuda 8, cuDNN 6, and Tensorflow 1. Use environment.yaml for anaconda and pip.txt for pip.

- Execute `bash ./patch.sh` to make resnet python files.

- Training is required for each protein.

- PyMol is also required to run. 

# Training

- Use makeTFtrain_pipeline.py to make tfrecord from training dataset (Glide docked results).
- Use train.sh to perform training.

# Inference

- Use makeTFtest_pipeline.py to make tfrecord from test dataset.
- Use test.sh to perform testing.

The image files or tfrecord file may be huge (around 1GB for 50 compounds), so prepare large disk or run separately.

# Reference

# License

This code refers open-source codes licensed under the Apache License 2.0.

resnet_utils.py and resnet_v2.py is licensed from https://github.com/tensorflow/models/

PyMol is published under a BSD-like license.
