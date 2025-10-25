#!/bin/bash
#SBATCH --job-name=fs_sam2_coseg_tiny
#SBATCH --output=/home/projects/u7633783/sam2/logs/fs_sam2_coseg_tiny_%j.out
#SBATCH --error=/home/projects/u7633783/sam2/logs/fs_sam2_coseg_tiny_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=u7633783@anu.edu.au
set -euo pipefail

export PYTHONPATH=/home/projects/u7633783/sam2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "[INFO] PYTHONPATH=${PYTHONPATH}"

SIF_PATH="/opt/apps/containers/conda/conda-nvidia-22.04-latest.sif"
SINGULARITY_BIND="/home/projects:/home/projects"

FS_ROOT="/home/projects/u7633783/sam2/FS-SAM2"

srun singularity exec --nv -B "${SINGULARITY_BIND}" "${SIF_PATH}" bash -lc "
  source ~/.venvs/fs-sam2/bin/activate;
  export PYTHONNOUSERSITE=1;
  export PYTHONPATH=/home/projects/u7633783/sam2;
  python -m torch.distributed.run --nproc_per_node 4 ${FS_ROOT}/train.py \
    --benchmark coco2017p \
    --datapath /home/projects/u7633783/datasets/coco \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_t.yaml \
    --sam2_ckpt /home/projects/u7633783/sam2/checkpoints/sam2.1_hiera_tiny.pt \
    --img_size 1024 --bsz 8 --epochs 50 --kshot 1 --nworker 6 \
    --exp_id coseg_tiny
"