import os
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
from itertools import chain

def walk_dir(dir_path):
    """ Walks a directory and returns all files with specific extensions. """
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    return file_paths

def process_files(file_paths):
    """ Organize files into a dictionary based on their directory structure. """
    frame_paths = {}
    for file_path in file_paths:
        # Extract the subdirectory and file to construct the key
        path_obj = Path(file_path)
        subdirectory = path_obj.parent.relative_to(base_dir_path)
        key = os.path.join(subdirectory.parts[0], subdirectory.parts[1])
        frame_paths.setdefault(key, []).append(path_obj.name)
    return frame_paths

def merge_dicts(dicts):
    """ Merge dictionaries by extending the lists of keys. """
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged.setdefault(k, []).extend(v)
    return merged

if __name__ == '__main__':
    base_dir_path = base_dir_path = Path("/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/preprocessing_data/kinetics_frames_256_3_12_train")
    output_pkl_path = base_dir_path / 'frame_paths.pkl'
    
    # Identify all subdirectories
    all_dirs = [d for d in base_dir_path.rglob('*') if d.is_dir()]

    # Walk directories in parallel and get all file paths
    with Pool(processes=cpu_count()) as pool:
        all_file_paths = list(tqdm(pool.imap_unordered(walk_dir, all_dirs), total=len(all_dirs), desc="Walking directories"))

    # Flatten the list of file paths
    all_file_paths = list(chain.from_iterable(all_file_paths))

    # Process files to organize them into a dictionary
    with Pool(processes=cpu_count()) as pool:
        dict_list = list(tqdm(pool.imap(process_files, [all_file_paths[i::cpu_count()] for i in range(cpu_count())]), total=cpu_count(), desc="Processing files"))

    # Merge dictionaries into one
    frame_paths = merge_dicts(dict_list)

    # Sort the frames for each key in the dictionary
    for key in tqdm(frame_paths, desc="Sorting frames"):
        frame_paths[key].sort()

    # Save the dictionary as a .pkl file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(frame_paths, f)

    print(f"Frame paths have been saved to {output_pkl_path}")
