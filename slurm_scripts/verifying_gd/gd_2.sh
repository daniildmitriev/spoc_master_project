#!/bin/bash
#SBATCH --chdir /home/dmitriev/spoc_master_project/slurm_output
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 4096
#SBATCH --time 20:00:00
#SBATCH -o testres.out
#SBATCH -e error.out
module load intel
module load python

declare -a manual_seed=(22)
declare -a n_samples=(1200 1600 2000)
declare -a n_features=(256)
declare -a n_hidden=(1)
declare -a lr=(0.0001 0.0002 0.0004)
declare -a n_epochs=(8000 4000 2000)
declare -a momentum=(0.0)
declare -a batch_size=(1)
for ((i=0;i<${#manual_seed[@]};++i)); do
for ((j=0;j<${#n_samples[@]};++j)); do
for ((k=0;k<${#n_features[@]};++k)); do
for ((l=0;l<${#n_hidden[@]};++l)); do
for ((m=0;m<${#lr[@]};++m)); do
for ((n=0;n<${#momentum[@]};++n)); do
srun python -u /home/dmitriev/spoc_master_project/main.py --root_dir verifying_gd \
 --n_train ${n_samples[j]} --n_test ${n_samples[j]} --n_features ${n_features[k]} --n_hidden ${n_hidden[l]} \
 --momentum_factor ${momentum[n]} --verbose_freq 100 --optimizer gd --n_epochs ${n_epochs[m]} --lr ${lr[m]} --threshold 1e-8
done
done
done
done
done
done
