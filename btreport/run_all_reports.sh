#!/bin/bash
#SBATCH --job-name=BTReport
#SBATCH --partition=ckpt
#SBATCH --account=kurtlab
#SBATCH --array=0-19
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/kurtlab/MSFT/vera/BTReport
#SBATCH --output=logs/generate/%A/btreport-%A_%a.out
#SBATCH --error=logs/generate/%A/btreport-%A_%a.err


echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Node          : $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Working dir   : $(pwd)"
echo "=========================================="


source ~/.bashrc
source /mmfs1/gscratch/kurtlab/MSFT/vera/BTReport/docs/btreport_paths.sh


module load apptainer
conda activate BTReport

# Start ollama server
# tmux new -d -s ollama_server \
#   "python3 -m btreport.ollama_server start-ollama --gpus 0,1"

# # Give the server a moment to come up
# sleep 15

# # Optional: sanity check
# tmux has-session -t ollama_server || {
#   echo "ERROR: Ollama tmux session failed to start"
#   exit 1
# }

# Interactive runs (salloc) have no array id; default to split 0.
SPLIT_NO="${SLURM_ARRAY_TASK_ID:-0}"

export PYTHONUNBUFFERED=1
python3 -m btreport.run_all_reports \
  --root_folder data_corebt \
  --num_splits 20 \
  --split_no "${SPLIT_NO}" \
  --run_name "5.4-mini" \
  # --merged_json /gscratch/kurtlab/MSFT/vera/BTReport/data_corebt/merged_reports_btreport_gpt_54_mini.json \
  --llm gpt-5.4-mini

echo "Array task ${SPLIT_NO} finished."