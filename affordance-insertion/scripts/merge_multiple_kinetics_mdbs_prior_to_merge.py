import os
import shutil

"""Just change DIR1,DIR2, and OUTPUTDIR below, it uses copy_db not delete or cut.
"""

def copy_db(src_dir, dest_dir, db_name):
    """ Copy the specified database folder to the destination. """
    src_path = os.path.join(src_dir, db_name)
    dest_path = os.path.join(dest_dir, db_name)
    if os.path.exists(src_path):
        shutil.copytree(src_path, dest_path)
        print(f'Copied {db_name} from {src_dir} to {dest_dir}')
    else:
        print(f'Warning: {src_path} does not exist.')

def process_chunk(chunk, dir1, dir2, output_dir):
    """ Process each chunk and copy required directories. """
    chunk_dir1 = os.path.join(dir1, chunk)
    chunk_dir2 = os.path.join(dir2, chunk)
    chunk_output_dir = os.path.join(output_dir, chunk)

    # Create output directory for the chunk if it doesn't exist
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Copy required databases
    copy_db(chunk_dir1, chunk_output_dir, 'clipmask_db')
    copy_db(chunk_dir1, chunk_output_dir, 'masks_db')
    copy_db(chunk_dir2, chunk_output_dir, 'frames_db')
    copy_db(chunk_dir2, chunk_output_dir, 'poses_db')

def main(dir1, dir2, output_dir):
    """ Main function to process all chunks and copy necessary directories. """
    chunks = [f'chunk_{i}' for i in range(8)]

    for chunk in chunks:
        process_chunk(chunk, dir1, dir2, output_dir)

if __name__ == '__main__':
    # Directories
    DIR1 = 'preprocessing_data/NTU_RGBD_frames_masks'
    DIR2 = 'preprocessing_data/NTU_RGBD_frames_pose'
    OUTPUT_DIR = 'preprocessing_data/NTU60_merged'

    # Run the main function
    main(DIR1, DIR2, OUTPUT_DIR)
