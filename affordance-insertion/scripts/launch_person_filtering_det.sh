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
#nvidia-smi
cd /srv/essa-lab/flash3/nwarner30/image_editing
source ~/.bashrc
source activate hallucinating-scenes_v3


# To Train
#INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kinetics_frames_256_3_14_train"
#INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/NTU_RGBD_frames"
INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/NTU_RGBD_120_ext_frames"


# Modify the OUTPUT_DIR to include a subfolder for each chunk
#BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_people_3_18_train"
#BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/NTU_RGBD_frames_people"
BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/NTU_RGBD_120_frames_people"



# Specify the chunk to process via environment variable
export CHUNK_NUM=7
export TOTAL_CHUNKS=8
export UNDERSAMPLING=1 # Choose 1 for original list.

# Create a subdirectory for the chunk
CHUNK_DIR="${BASE_OUTPUT_DIR}/chunk_${CHUNK_NUM}"
#mkdir -p "${CHUNK_DIR}"

# Uncomment the below lines if you want to run the script for validation
#INPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kinetics_frames_256_3_12_val"
#BASE_OUTPUT_DIR="/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kinetics_people_3_18_val"

# Execute the Python script with the specified input and output directories
python hallucinating-scenes/scripts/other_scripts_BLIP_or_original_repo/\
data_filter_people.py input_dir=${INPUT_DIR} output_dir=${CHUNK_DIR} chunk_num=${CHUNK_NUM} total_chunks=${TOTAL_CHUNKS}
