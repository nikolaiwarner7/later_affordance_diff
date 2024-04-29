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

# Base directories
# Kinetics
#BASE_INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_pose_full_4_11"
#BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_masks_full_4_11"


#NTU60
#BASE_INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/NTU_RGBD_frames_pose"
#BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/NTU_RGBD_frames_masks"

#NTU120
BASE_INPUT_DIR='/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/NTU_RGBD_120_frames_pose'
BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/NTU_RGBD_120_frames_masks"

# Loop over the chunk indices
for i in {0..7}; do
    echo "Processing mask generation for chunk $i..."
    
    INPUT_DIR="${BASE_INPUT_DIR}/chunk_$i"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/chunk_$i"
    
    # Create output directory if it doesn't exist
    #mkdir -p "${OUTPUT_DIR}"
    
    # Execute the Python script with the specified input and output directories for each chunk
    
    python affordance-insertion/ldm/data/data_generate_masks.py input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR}
    
    echo "Finished mask generation for chunk $i."
done
