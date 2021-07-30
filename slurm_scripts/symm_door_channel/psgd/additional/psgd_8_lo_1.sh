#!/bin/bash
#SBATCH --chdir /home/dmitriev/spoc_master_project/slurm_output
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 4096
#SBATCH --time 30:00:00
#SBATCH -o testres.out
#SBATCH -e error.out
module load intel
module load python

declare -a manual_seed=(22)
declare -a sample_complexity=(8.0)
declare -a n_features=(1024)
declare -a n_hidden=(1)
declare -a lr=(0.01)
declare -a n_epochs=(2000)
declare -a momentum=(0.0)
declare -a batch_size=(128)
declare -a optimizer=("p-sgd")
declare -a tau=(1)
declare -a K=(0.67449)
declare -a activation=("symmetric-door")
declare -a loss=("logloss")
for ((s=0;s<${#manual_seed[@]};++s)); do
for ((a=0;a<${#sample_complexity[@]};++a)); do
for ((f=0;f<${#n_features[@]};++f)); do
for ((h=0;h<${#n_hidden[@]};++h)); do
for ((l=0;l<${#lr[@]};++l)); do
for ((m=0;m<${#momentum[@]};++m)); do
for ((o=0;o<${#optimizer[@]};++o)); do
for ((b=0;b<${#batch_size[@]};++b)); do
for ((t=0;t<${#tau[@]};++t)); do
for ((k=0;k<${#K[@]};++k)); do
for ((c=0;c<${#activation[@]};++c)); do
for ((q=0;q<${#loss[@]};++q)); do
srun python -u /home/dmitriev/spoc_master_project/main.py --root_dir symmetric_door --start_seed ${manual_seed[s]} \
 --batch_size ${batch_size[b]} --sample_complexity ${sample_complexity[a]} --n_test 100 \
 --n_features ${n_features[f]} --n_hidden ${n_hidden[h]} --momentum_factor ${momentum[m]} \
 --verbose_freq 100 --optimizer ${optimizer[o]} --n_epochs ${n_epochs[l]} --lr ${lr[l]} --train_threshold 1e-8 \
 --gradient_threshold 5e-10 --test_threshold 1e-3 --n_runs 10 --persistence_time ${tau[t]} \
 --loss ${loss[q]} --activation ${activation[c]} --symmetric_door_channel_K ${K[k]}
done
done
done
done
done
done
done
done
done
done
done
done
