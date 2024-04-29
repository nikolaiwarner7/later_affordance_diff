#!/bin/bash
#SBATCH -p 'essa-lab'
#SBATCH -c 36
#SBATCH -G 'a40:4'
#SBATCH --exclude nestor
#SBATCH --nodes 1
#SBATCH --qos 'short'
#SBATCH -o '/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/slurm_runs/output_%j.txt'
#SBATCH -e '/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/slurm_runs/error_%j.txt'

export PYTHONUNBUFFERED=TRUE
cd /srv/essa-lab/flash3/nwarner30/image_editing
source ~/.bashrc
source activate hallucinating-scenes_v3

# Set the dataset name
DATASET="chunk" # Specifically processed w pre-merge script see doc for details

# Set base input and output directories
BASE_INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_full_4_11_to_merge"

# Set the new output path
OUTPUT_PATH="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/final_merged_kinetics_full_4_11_v2"
mkdir -p ${OUTPUT_PATH}

# Run SCRIPT3 for each split
for SPLIT in 0 1; do
    echo "Merging LMDB for split $SPLIT..."
    
    # Call SCRIPT3 with the dataset name, input directory, and split index
    python affordance-insertion/ldm/data/data_merge_lmdb.py ${DATASET} ${BASE_INPUT_DIR} ${SPLIT}
    
    echo "Finished merging LMDB for split $SPLIT."
done