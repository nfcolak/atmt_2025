#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=8:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=stopping_criteria.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Experiment comparing relative threshold pruning (rp) and relative local threshold pruning (rpl)
# The threshold values are taken from Table 1 in the paper "Beam Search Strategies for Neural Machine Translation"
echo "Translation with relative threshold pruning at rp=0.6"
python translate.py \
    --cuda \
    --input  toy_example/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output_rp0.6.txt \
    --max-len 128 \
    --beam-size 5 \
    --rp 0.6

echo "Translation with relative local threshold pruning at rpl=0.02"
python translate.py \
    --cuda \
    --input  toy_example/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output_rpl0.02.txt \
    --max-len 128 \
    --beam-size 5 \
    --rpl 0.02

echo "Translation with rpl=0.6 and rpl=0.02"
python translate.py \
    --cuda \
    --input  toy_example/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output_rp0.6_rpl0.02.txt \
    --max-len 128 \
    --beam-size 5 \
    --rp 0.6 \
    --rpl 0.02