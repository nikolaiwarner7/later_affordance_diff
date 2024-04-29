"""
import pickle
from pathlib import Path

# Define the path to the frame_paths.pkl file
pkl_path = Path("/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_frames_256_3_12_val/frame_paths.pkl")

# Function to load and display the first few entries of the pkl file
def load_and_display(pkl_file_path, num_entries=2):
    # Ensure the file exists
    if not pkl_file_path.is_file():
        print(f"File not found: {pkl_file_path}")
        return

    # Load the pickle file
    with open(pkl_file_path, "rb") as file:
        data = pickle.load(file)

    # Display the first few entries
    for i, (key, value) in enumerate(data.items()):
        if i >= num_entries: break
        print(f"Key: {key}, Value: {value}\n")

if __name__ == "__main__":
    load_and_display(pkl_path)
"""
#import tqdm
from typing import Any, Callable, List, Optional, Tuple
import joblib
import os
import sys
import tqdm 

import pickle
from pathlib import Path


class ParallelProgressBar(joblib.Parallel):
    def tqdm(self, **kwargs):
        self._tqdm_kwargs = kwargs

    def __call__(self, function: Callable[[int, Any], Any], inputs: List):
        tqdm_kwargs = getattr(self, "_tqdm_kwargs", dict())

        tqdm_kwargs["total"] = tqdm_kwargs.get("total", len(inputs))
        tqdm_kwargs["dynamic_ncols"] = tqdm_kwargs.get("dynamic_ncols", True)

        with tqdm.tqdm(**tqdm_kwargs) as self._progress_bar:

            return super().__call__(
                joblib.delayed(function)(index, input)
                for index, input in enumerate(inputs)
            )

    def print_progress(self):
        self._progress_bar.n = self.n_completed_tasks
        self._progress_bar.refresh()



#import utils

# Assuming utils.py is in the same directory as this script, or the path to it is correctly appended to sys.path

# Define the base directory
base_dir_path = Path("/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_frames_256_3_12_train")

# Number of workers for multiprocessing
NUM_WORKERS = 4  # Adjust this number based on your machine's capability

# This function will be the worker that processes each directory
def process_directory(directory):
    frame_paths = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a frame image
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                # Extract the relative directory name and file to construct the key
                relative_dir = os.path.relpath(root, start=base_dir_path)
                key = os.path.join(relative_dir, file)
                # Append the file to the corresponding list in the dictionary
                if relative_dir not in frame_paths:
                    frame_paths[relative_dir] = []
                frame_paths[relative_dir].append(file)
    
    # Return a tuple (directory, sorted frame list)
    return relative_dir, sorted(frame_paths[relative_dir])

# Main logic for using multiprocessing to walk through directories and process them
if __name__ == "__main__":
    # Generate a list of directories to process
    directories_to_process = [x for x in base_dir_path.iterdir() if x.is_dir()]
    
    # Use the utils.ParallelProgressBar for multiprocessing with a progress bar
    with ParallelProgressBar(n_jobs=NUM_WORKERS) as parallel:
        parallel.tqdm(desc="Processing Directories", unit="directory")
        # Note that the parallel processing function should take the index and directory as arguments
        results = parallel(process_directory, directories_to_process)

    # Combine the results from the workers
    frame_paths = dict(results)

    # Define the path to save the pickle file
    output_pkl_path = base_dir_path / 'frame_paths.pkl'

    # Save the dictionary as a .pkl file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(frame_paths, f)

    print(f"Frame paths have been saved to {output_pkl_path}")

