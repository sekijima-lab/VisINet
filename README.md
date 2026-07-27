# Description

VisINet: automating visual inspection for protein-ligand docking results

The source code is an image-based virtual screening method named VisINet (VISual Inspection Network), where compounds were screened solely based on images taken from protein-ligand docking pose and known activity of ligands, without any other explicit information of ligands, such as molecular structures or docking scores.

# Installation

- This model is tested on linux, with python 3.5, cuda 8, cuDNN 6, and Tensorflow 1. Use environment.yaml for anaconda and pip.txt for pip.
- A `Dockerfile` is provided as a reproducible alternative to setting up the conda environment by hand (Python 3.7 / CUDA 10.0 / cuDNN 7.6.5 / TensorFlow 1.15, matching environment.yaml/pip.txt). Build with `docker build -t visinet .`; `patch.sh` runs automatically as part of the image build.

- Execute `bash ./patch.sh` to make resnet python files.

- Training is required for each protein.

- PyMol is also required to run. 

# Training

- Use makeTFtrain_pipeline.py to make tfrecord from training dataset (Glide docked results).
- Use train.sh to perform training.
- `mnist3.py` and `test4.py` take the protein name as their first argument (it must match one of the functions defined in `ratio.py`); see the usage docstring at the top of each file.

# Inference

- Use makeTFtest_pipeline.py to make tfrecord from test dataset.
- Use test.sh to perform testing.
- `test4.py` writes per-compound scores to a file but does not compute AUC/EF1% itself. `eval_test.py` is provided to reproduce the paper's compound-level evaluation protocol (3D average pooling across a compound's rendered views, then AUC and EF1% via scikit-learn) directly against a trained checkpoint. Usage: `python3 eval_test.py <protein_name> <model_dir> <tfrecord> [<tfrecord> ...]`.

The image files or tfrecord file may be huge (around 1GB for 50 compounds), so prepare large disk or run separately.

# Reference

# License

This code **except below ** is published as MIT license.

resnet_utils.py and resnet_v2.py (original files, before applying patch) are licensed at https://github.com/tensorflow/models/ .

PyMol is published under a BSD-like license at https://github.com/schrodinger/pymol-open-source .
