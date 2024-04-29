import json

# Load the original dictionary
with open('train_captions_frame_mapping_gpu_v2.json', 'r') as f:
    original_dict = json.load(f)

# Initialize the new dictionaries
clip_to_caption = {}
clip_to_frame_indices = {}

# Process each item in the original dictionary
for caption_key, frames_list in original_dict.items():
    # Extract CAPTION_NUM from the caption_key
    caption_num = caption_key.split('_')[-1].split('.')[0]

    # Extract CLIP_NUM, REF_FRAME_IDX, and TARGET_FRAME_IDX from the first item in frames_list
    clip_num = int(frames_list[0].split('_')[1])
    ref_frame_idx = int(frames_list[0].split('_')[-1])  # Convert to int to remove leading zeros
    target_frame_idx = int(frames_list[1].split('_')[-1])  # Convert to int to remove leading zeros

    # Update the new dictionaries
    clip_to_caption[clip_num] = caption_num
    clip_to_frame_indices[clip_num] = [ref_frame_idx, target_frame_idx]

# Save the new dictionaries as JSON
with open('train_clip_to_caption.json', 'w') as f:
    json.dump(clip_to_caption, f)

with open('train_clip_to_frame_indices.json', 'w') as f:
    json.dump(clip_to_frame_indices, f)

print("Dictionaries saved successfully.")
