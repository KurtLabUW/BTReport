#!/bin/bash
#SBATCH --job-name=apply_mni
#SBATCH --account=kurtlab
#SBATCH --partition=ckpt
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/gscratch/kurtlab/MSFT/metadata_analysis/register_mni/apply_logs/out_apply_%A_%a.txt
#SBATCH --error=/gscratch/kurtlab/MSFT/metadata_analysis/register_mni/apply_logs/err_apply_%A_%a.txt
#SBATCH --chdir=/gscratch/kurtlab/MSFT/metadata_analysis/register_mni

module load apptainer
export APPTAINER_CACHEDIR=/gscratch/kurtlab/apptainer/cache
export APPTAINER_TMPDIR=/gscratch/kurtlab/apptainer/tmp
export SUBJECTS_DIR="/gscratch/kurtlab"

PATHS_FILE="/gscratch/kurtlab/MSFT/metadata_analysis/register_mni/paths_apply_transform.csv"
TOTAL_SPLITS=10

echo "[$(date)] Starting SynthMorph apply split $SLURM_ARRAY_TASK_ID / $((TOTAL_SPLITS-1))"

python3 -u register_split_apply_transform.py \
    "$PATHS_FILE" "$SLURM_ARRAY_TASK_ID" "$TOTAL_SPLITS"

echo "[$(date)] Finished apply split $SLURM_ARRAY_TASK_ID"


# python3 -u register_split_apply_transform.py \
#     "$PATHS_FILE" 0 "$TOTAL_SPLITS"