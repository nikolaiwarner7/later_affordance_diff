import os
import json
import sys
from PIL import Image
import imagehash
sys.path.append("/coc/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic/")
sys.path.append("/coc/flash3/nwarner30/image_editing/")
from database import  Database, ImageDatabase
#from  /srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic import Database, ImageDatabase
import ipdb

# Directory where composite images are stored
composite_dir = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kulal_kinetics_composites_train"
# Output JSON file path
output_json = "train_captions_frame_mapping.json"

# Initialize the dictionary
video_frames_mapping = {}

frames_db_path = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/training_data/kinetics/frames_db'
frames_db = ImageDatabase(
    frames_db_path, readahead=False, lock=False
)

clips_db_path = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/training_data/kinetics/clipmask_db'
clips_db = Database(clips_db_path, lock=False)

keys = []

clip_keys = clips_db.keys()
for clip_key in clip_keys:
    frame_keys = clips_db[clip_key]
    if len(frame_keys) >= 2:
        keys.append(frame_keys)
# is a list of lists

THRESHOLD = 5


prefixes = [clips[0].split("_frame")[0] for clips in keys]
# Now, `prefixes` contains all unique clip identifiers without the frame part

def find_frame_index(frame, frames_db, frame_num):
    # Compute hash of the input frame for comparison
    frame_hash = imagehash.average_hash(frame)

    # get list of possible frames
    #ipdb.set_trace()
    frame_num_formatted = "clip_" + str(frame_num).zfill(7)
    video_idx = prefixes.index(frame_num_formatted)
    possible_frames = keys[video_idx]

    for i, pframe in enumerate(possible_frames):
        # compare frame, pframe
        # Loads in PIL format
        ipdb.set_trace()
        pframe_img = frames_db[pframe]
        pframe_hash = imagehash.average_hash(pframe_img)

        # Compare hashes
        if frame_hash - pframe_hash < THRESHOLD:  # Define a suitable threshold
            return i


# Loop through each composite image in the directory
for filename in os.listdir(composite_dir):
    if filename.startswith("composite_") and filename.endswith(".jpg"):
        frame_num = filename[len("composite_"):-len(".jpg")]

        # 1) load the composite image, separate it into left, right
        # 2) match the reference image(left), and target image (right) with 
            # the images in the frames_Db at the same frame, get the index number (0-5) for each
        # 3) save to json dict/

        ipdb.set_trace()

        # Load the composite image
        composite_path = os.path.join(composite_dir, filename)
        composite_image = Image.open(composite_path)
        
        # Split the composite image into left (reference) and right (target) images
        width, height = composite_image.size
        left_image = composite_image.crop((0, 0, width // 2, height))
        right_image = composite_image.crop((width // 2, 0, width, height))
        
        # Find the index of each frame in the frames_db
        ref_index = find_frame_index(left_image, frames_db, frame_num)
        target_index = find_frame_index(right_image, frames_db, frame_num)
        
        # Update the mapping dictionary
        video_frames_mapping[frame_num] = [ref_index, target_index]
        


# Serialize the dictionary to JSON
with open(output_json, 'w') as f:
    json.dump(video_frames_mapping, f, indent=4)

