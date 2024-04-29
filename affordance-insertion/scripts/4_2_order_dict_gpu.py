import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import ipdb

# Assuming you've already added necessary paths to sys.path
import sys
sys.path.append("/coc/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic/")
sys.path.append("/coc/flash3/nwarner30/image_editing/")
from database import ImageDatabase, Database

# Configuration
composite_dir = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kulal_kinetics_composites_train"
output_json = "train_captions_frame_mapping_gpu_v2.json"
frames_db_path = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/training_data/kinetics/frames_db'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the pre-trained model and transformation
model = models.resnet50(pretrained=True).to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_feature_vector(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

# Initialize databases
frames_db = ImageDatabase(frames_db_path, readahead=False, lock=False)

ipdb.set_trace() # Can use this just to search db without using rest of script

# Extract feature vectors for each frame in the frames database
print("Extracting feature vectors for database images...")
db_features = []
db_frame_ids = list(frames_db.keys())  # Assuming this returns a list of identifiers
for frame_id in tqdm(db_frame_ids):
    frame = frames_db[frame_id]  # Load your frame as PIL image
    features = get_feature_vector(frame)
    db_features.append(features)
db_features = np.array(db_features).astype('float32')


# Initialize FAISS index
index = faiss.IndexFlatL2(db_features.shape[1])
if device == 'cuda':
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
index.add(db_features)

# Function to find closest frame in database
def find_closest_frame(feature_vector, index):
    distances, indices = index.search(feature_vector.reshape(1, -1), 1)
    return db_frame_ids[indices.flatten()[0]]

# Process composite images
video_frames_mapping = {}
for filename in tqdm(os.listdir(composite_dir)):
    if not filename.startswith("composite_") or not filename.endswith(".jpg"):
        continue
    composite_path = os.path.join(composite_dir, filename)
    composite_image = Image.open(composite_path)
    
    width, height = composite_image.size
    left_image = composite_image.crop((0, 0, width // 2, height))
    right_image = composite_image.crop((width // 2, 0, width, height))
    
    left_feature = get_feature_vector(left_image)
    right_feature = get_feature_vector(right_image)
    
    left_index = find_closest_frame(left_feature, index)
    right_index = find_closest_frame(right_feature, index)
    
    video_frames_mapping[filename] = [left_index, right_index]

# Save mappings to JSON
with open(output_json, 'w') as f:
    json.dump(video_frames_mapping, f)

print("Mapping completed and saved.")
