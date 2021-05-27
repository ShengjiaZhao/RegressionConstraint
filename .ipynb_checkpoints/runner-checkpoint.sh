#!/bin/bash
#SBATCH --partition=atlas
#SBATCH --time=150:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --job-name=gdp_runner
#SBATCH --output=/atlas/u/chenlin/logs/stdout/%j.out
#SBATCH --error=/atlas/u/chenlin/logs/stderr/%j.err
#SBATCH --exclude=atlas6,atlas1,atlas2,atlas3,atlas5,atlas4,atlas13

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS


#python sustain_regression.py --num_run 10
#python sustain_regression.py --train_calib --num_run 10
#python sustain_regression.py --train_bias_y --num_run 10
#python sustain_regression.py --train_bias_f --num_run 10
#python sustain_regression.py --train_bias_y --train_calib --num_run 10
#python sustain_regression.py --train_bias_y --train_calib --train_bias_f --num_run 10
#python sustain_regression.py --train_bias_f --train_calib --num_run 10

#python sustain_regression.py --num_run 10 --dataset poverty
#python sustain_regression.py --train_calib --num_run 10 --dataset poverty
#python sustain_regression.py --train_bias_y --num_run 10 --dataset poverty --num_bins 5
#python sustain_regression.py --train_bias_f --num_run 10 --dataset poverty --num_bins 5
#python sustain_regression.py --train_bias_y --train_calib --num_run 10 --dataset poverty --num_bins 5
#python sustain_regression.py --train_bias_y --train_calib --train_bias_f --num_run 10 --dataset poverty --num_bins 5
#python sustain_regression.py --train_bias_f --train_calib --num_run 10 --dataset poverty --num_bins 5


#python sustain_utility.py --run_label 0
#python sustain_utility.py --train_calib --run_label 0
#python sustain_utility.py --train_bias_y --run_label 0
#python sustain_utility.py --train_bias_f --run_label 0
#python sustain_utility.py --train_bias_y --train_calib --run_label 0
#python sustain_utility.py --train_bias_y --train_calib --train_bias_f --run_label 0
#python sustain_utility.py --train_bias_f --train_calib --run_label 0

# KNN results
#python sustain_regression.py --dataset poverty --num_bins 0 --num_run 10 --model 'linear'
#python sustain_regression.py --train_calib --num_run 10 --num_bins 0 --dataset poverty --model 'linear'
#python sustain_regression.py --train_bias_y --num_run 10 --dataset poverty --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_f --num_run 10 --dataset poverty --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_y --train_calib --num_run 10 --dataset poverty --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_y --train_calib --train_bias_f --num_run 10 --dataset poverty --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_f --train_calib --num_run 10 --dataset poverty --num_bins 0 --model 'linear'

#python sustain_regression.py --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_calib --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_y --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_f --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_y --train_calib --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_y --train_calib --train_bias_f --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_f --train_calib --num_run 10 --num_bins 0 --model 'linear'
#python sustain_regression.py --train_bias_y --train_bias_f --num_run 10 --num_bins 0 --model 'linear'


#python sustain_regression_original.py --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_calib --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_bias_y --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_bias_f --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_bias_y --train_calib --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_bias_y --train_calib --train_bias_f --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_bias_f --train_calib --num_run 10 --num_bins 0 --model 'small' --seed 20
#python sustain_regression_original.py --train_bias_y --train_bias_f --num_run 10 --num_bins 0 --model 'small' --seed 20

#python sustain_regression.py --model linear --re_calib
#python sustain_regression.py --model linear --re_bias_f
#python sustain_regression.py --model linear --re_bias_y
