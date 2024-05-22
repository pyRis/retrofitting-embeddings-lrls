#!/bin/bash 

pip install h5py
python train.py --first-step-only

python train.py --second-step-only