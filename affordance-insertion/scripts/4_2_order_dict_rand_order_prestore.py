import os
import json
import sys
from PIL import Image
import imagehash
from multiprocessing import Pool, cpu_count
sys.path.append("/coc/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic/")
sys.path.append("/coc/flash3/nwarner30/image_editing/")
from database import  Database, ImageDatabase
#from  /srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic import Database, ImageDatabase
import ipdb
from tqdm import tqdm
# Directory where composite images are stored
composite_dir = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kulal_kinetics_composites_train"
# Output JSON file path
output_json = "train_captions_frame_mapping_perceptual.json"

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

# Precompute hashes for all frames in frames_db
print("Precomputing frame hashes...")

def compute_hash(frame_identifier):
    """Compute the hash for a single frame."""
    db_frame = frames_db[frame_identifier]  # Assuming this returns a PIL Image
    db_frame_hash = imagehash.phash(db_frame)
    return frame_identifier, db_frame_hash

# Precompute hashes for all frames in frames_db using multiprocessing
def precompute_hashes_parallel():
    frame_identifiers = list(frames_db.keys())
    pool = Pool(processes=cpu_count())  # Or set manually to 20 or desired number of cores
    with pool:
        results = list(tqdm(pool.imap(compute_hash, frame_identifiers), total=len(frame_identifiers)))
    return dict(results)

frame_hashes = precompute_hashes_parallel()



# Initialize the dictionary for video frames mapping
video_frames_mapping = {}

# Adjusted threshold for hash comparison
THRESHOLD = 5

def find_frame_index(frame, frame_hashes):
    frame_hash = imagehash.phash(frame)
    min_distance = float('inf')
    closest_frame_index = None
    
    for frame_identifier, db_frame_hash in frame_hashes.items():
        hash_distance = frame_hash - db_frame_hash
        if hash_distance < min_distance:
            min_distance = hash_distance
            closest_frame_index = frame_identifier
        if hash_distance < THRESHOLD:
            #ipdb.set_trace()
            break
    
    return closest_frame_index

# Process each composite image
print("Processing composite images...")
for filename in tqdm(os.listdir(composite_dir)):
    if filename.startswith("composite_") and filename.endswith(".jpg"):
        composite_path = os.path.join(composite_dir, filename)
        composite_image = Image.open(composite_path)
        
        width, height = composite_image.size
        left_image = composite_image.crop((0, 0, width // 2, height))
        right_image = composite_image.crop((width // 2, 0, width, height))
        
        ref_index = find_frame_index(left_image, frame_hashes)
        target_index = find_frame_index(right_image, frame_hashes)
        
        video_frames_mapping[filename] = [ref_index, target_index]

# Save the mapping to a JSON file
with open(output_json, 'w') as f:
    json.dump(video_frames_mapping, f, indent=4)
