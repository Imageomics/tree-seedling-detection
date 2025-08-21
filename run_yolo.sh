#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=72:00:00
#SBATCH --mem=30gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhou.m@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1



source /blue/azare/zhou.m/anaconda3/bin/activate deepforest

python train_yolo.py --batch_size $1 --modality $2 --epoch $3
