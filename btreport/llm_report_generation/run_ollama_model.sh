
export PATH="${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin"

# Model directory
export APPTAINERENV_OLLAMA_MODELS="/pscratch/sd/j/jehr/ollama/ollama_models"

# Path to image
IMAGE="/pscratch/sd/j/jehr/ollama/ollama.sif"

apptainer exec --nv \
   -B /pscratch:/pscratch \
   -B /cvmfs:/cvmfs \
   $IMAGE \
   ollama run symptoma/medgemma3:27b
