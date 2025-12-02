#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=8:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment1.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Experiment with alphas between 0 and 1
for alpha in 0 0.25 0.5 0.75 1
do
    echo "Translation with length norm alpha=$alpha"
    # TRANSLATE
    python translate.py \
        --cuda \
        --input  toy_example/data/raw/test.cz \
        --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
        --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
        --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
        --output cz-en/output${alpha}.txt \
        --max-len 128 \
        --beam-size 5 \
        --alpha $alpha
done