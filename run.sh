#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=72:00:00
#$-j y
#$-cwd
#$-o temp_output.txt
mv temp_output.txt /home/acf15412ed/parker-preprocessing-comparison/command_outputs/$1_$2_$3.txt

source /etc/profile.d/modules.sh
source /home/acf15412ed/train-venv/bin/activate
module load gcc/12.2.0 python/3.11/3.11.2
module load cuda/11.6/11.6.2 cudnn/8.6/8.6.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/11.6.2/

if $2 == "fusion"; then
  python3.11 /home/acf15412ed/parker-preprocessing-comparison/src/train/fusion/train.py $1 $3
elif $2 == "rgb_concat"; then
  python3.11 /home/acf15412ed/parker-preprocessing-comparison/src/train/rgb_concat/train.py $1 $3
elif $2 == "multihead"; then
  python3.11 /home/acf15412ed/parker-preprocessing-comparison/src/train/multihead/train.py $1 $3
elif $2 == "singlehead"; then
  python3.11 /home/acf15412ed/parker-preprocessing-comparison/src/train/singlehead/train.py $1 $3
else
  echo "Invalid arugment given!"
fi