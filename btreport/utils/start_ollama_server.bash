#!/bin/bash

GPUS="0"   # Default if none provided

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"; shift 2;;
        *)
            echo "Unknown option: $1"; exit 1;;
    esac
done

echo "OLLAMA GPUs set to: $GPUS"
export CUDA_VISIBLE_DEVICES=$GPUS



# Load Apptainer 
export PATH="${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin"

# Model directory
export APPTAINERENV_OLLAMA_MODELS="/pscratch/sd/j/jehr/ollama/ollama_models"

# Path to image
IMAGE="/pscratch/sd/j/jehr/ollama/ollama.sif"

# Print node and GPU info 
echo "Node: $(hostname)"
echo "GPUs Available:"
nvidia-smi | sed 's/^/    /'

echo "Starting Ollama server..."
echo "Press Ctrl+C to stop."

# Start ollama server
apptainer exec --nv \
    -B /pscratch:/pscratch \
    -B /cvmfs:/cvmfs \
    "$IMAGE" ollama serve
