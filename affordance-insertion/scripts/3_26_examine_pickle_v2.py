import os
import pickle
from pathlib import Path
from tqdm import tqdm

# Define the directory to walk through
base_dir_path = Path("/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_frames_256_3_12_train")

# Initialize an empty dictionary to hold the frame paths
frame_paths = {}

# Walk through the directory structure
for root, dirs, files in tqdm(os.walk(base_dir_path), desc="Walking directories"):
    for file in files:
        # Check if the file is a frame image
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            # Extract the subdirectory name and file to construct the key
            subdirectory = Path(root).relative_to(base_dir_path)
            key = os.path.join(subdirectory.parts[0], subdirectory.parts[1])  # Assuming the structure is as described
            # Append the file to the corresponding list in the dictionary
            if key not in frame_paths:
                frame_paths[key] = []
            frame_paths[key].append(file)

# Sort the frames for each key
for key in tqdm(frame_paths, desc="Sorting frames"):
    frame_paths[key].sort()

# Define the path to save the pickle file
output_pkl_path = base_dir_path / 'frame_paths.pkl'

# Save the dictionary as a .pkl file
with open(output_pkl_path, 'wb') as f:
    pickle.dump(frame_paths, f)

print(f"Frame paths have been saved to {output_pkl_path}")
