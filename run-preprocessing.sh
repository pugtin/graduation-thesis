#$-o temp_output.txt
#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=72:00:00
#$-j y
#$-cwd
#$-o temp_output.txt

source /etc/profile.d/modules.sh
source /home/acf15412ed/train-venv/bin/activate
module load gcc/12.2.0 python/3.11/3.11.2
module load cuda/11.6/11.6.2 cudnn/8.6/8.6.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/11.6.2/

# python 3.11 ~ wisdm gasf(or gadf or mtf or fusion1)
python3.11 /home/acf15412ed/parker-preprocessing-comparison/src/preprocessing/composer.py $1 $2