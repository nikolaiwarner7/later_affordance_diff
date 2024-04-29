import os
import sys
sys.path.append("/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/")
sys.path.append("/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/")
import base64
import requests
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
import ipdb
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


DATASET_NAME = 'Kinetics'
#DATASET_NAME = random.choice(['Kinetics', 'HVU'])
# Choose which one

# Constants
SPLIT = 'train'
#SPLIT = 'val'

OUTPUT_PATH = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/"

# the config file including information about the data module
config_filename = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/configs/latent-diffusion/affordance.yaml"

# switch to api key if count exceeds certain limit

NUM_SHOTS = 10 # string
NUM_EXAMPLES = 20
api_key_count_limit = 5  
USE_EXISTING_CAPTIONS = True
# Rate limits are 500 + now. Request increase on acct.
PRINT_VERBOSE_RESPONSE = True

# Define your API keys
from API_Keys import gt_api_key, nw_api_key
api_key = nw_api_key
iteration_count = 0


TEXT_PROMPT = 'There are two frames in the image. Caption the difference between frames (assume left or top image comes first).\
    Pay attention only to pose and objects interacted with. If no significant change, state so. Follow the previous examples.'



if DATASET_NAME == 'Kinetics':
    OUTPUT_DIR = os.path.join(OUTPUT_PATH,'kulal_kinetics_composites_%s/' % SPLIT)
    OUTPUT_JSON = os.path.join(OUTPUT_PATH, 'kulal_kinetics_composites_%s.json' % SPLIT)
    # Load the captions
    json_path = "/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/data/example_captions_for_gpt/example_captions.json"
    captions_dir = "/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/data/example_captions_for_gpt/"

elif DATASET_NAME == 'HVU':  # HVU
    OUTPUT_DIR = os.path.join(OUTPUT_PATH,'kulal_hvu_composites_%s/' % SPLIT)
    OUTPUT_JSON = os.path.join(OUTPUT_PATH, 'kulal_hvu_composites_%s.json' % SPLIT)
    # Load the captions
    # Change these for HVU specific later
    json_path = "/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/data/example_captions_for_gpt/example_captions.json"
    captions_dir = "/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/data/example_captions_for_gpt/"
else:
    print("Error, select Kinetics or HVU.")

###############################################
# NOTHING TO MODIFY PAST HERE JUST TO RUN ##
###############################################

config = OmegaConf.load(config_filename)
data_config = config.data

# Setup your data
data = instantiate_from_config(data_config)

data.setup()

ipdb.set_trace()

# Setup Train DataLoader and Validation DataLoader 
# based on train/val SPLIT
BATCH_SIZE = 4
data.batch_size = BATCH_SIZE

if SPLIT=='train':
    dataloader = DataLoader(data.datasets["train"], batch_size=data.batch_size,
                              num_workers=data.num_workers, shuffle=True)
elif SPLIT == 'val':
    dataloader = DataLoader(data.datasets["validation"], batch_size=data.batch_size,
                            num_workers=data.num_workers, shuffle=False)
else:
    print("Error loading dataloader, check SPLIT.")


# Function to create and save composite image (same as in REF script)
def create_composite_image(image1_tensor, image2_tensor, output_dir, output_image_name, max_dim=512):
    device = image1_tensor.device  # Get the device of the input tensors

    image1 = ToPILImage()(image1_tensor.cpu().permute(2, 0, 1))
    image2 = ToPILImage()(0.5 * (image2_tensor.cpu().permute(2, 0, 1) + 1))

    images = [image1, image2]
    
    # Determine the total width and height for the composite image
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    total_height = sum(heights)

    # Check if the composite should be wide or tall
    if widths[0] > heights[0]:  # Wide composite
        max_width = max(widths)
        composite_image = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            composite_image.paste(im, (0, y_offset))
            y_offset += im.height
    else:  # Tall composite
        max_height = max(heights)
        composite_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            composite_image.paste(im, (x_offset, 0))
            x_offset += im.width

    # Resize the composite image
    if composite_image.width > composite_image.height:
        new_height = int(composite_image.height * max_dim / composite_image.width)
        composite_image = composite_image.resize((max_dim, new_height))
    else:
        new_width = int(composite_image.width * max_dim / composite_image.height)
        composite_image = composite_image.resize((new_width, max_dim))

    # Pad the composite image to make it square
    final_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    x_center = (max_dim - composite_image.width) // 2
    y_center = (max_dim - composite_image.height) // 2
    final_image.paste(composite_image, (x_center, y_center))

    # Construct the output image name and path, and save the final composite image
    output_image_path = os.path.join(output_dir, output_image_name)
    
    final_image.save(output_image_path)

    return output_image_path

def make_batch_request(encoded_images, exemplar_images=None, exemplar_captions=None, max_examples=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = []

    # Append exemplar messages
    if exemplar_images and exemplar_captions and len(exemplar_images) == len(exemplar_captions):
        num_examples = len(exemplar_images) if max_examples is None else min(max_examples, len(exemplar_images))
        messages.append({
            "role": "system",
            "content": "I will provide examples of scene difference captions given composite images (of 2 different scenes from the same video)\
                    There are two frames in the image. Caption the difference between frames (assume left or top image comes first)."
        })
        for i in range(num_examples):
            messages.append({
                "role": "user",
                "content": exemplar_captions[i]
            })
        messages.append({
            "role": "system",
            "content": "I have provided examples of scene difference captions given composite images (of 2 different scenes from the same video)\
                    There are two frames in the image. Caption the difference between frames (assume left or top image comes first)."
        })
    # Append messages for each of the batch images to be captioned
    for encoded_image in encoded_images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": TEXT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}}
            ]
        })

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 2000  # Adjust as needed
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_batch(batch, exemplar_images, exemplar_captions, num_examples, api_requests, is_training):
    global iteration_count
    global api_key

    # process the batch
    for i in range(len(batch)):
        # make composite image
        
        composite_image_name = "composite_%s.jpg" % batch['data_idx'][i]
        if composite_image_name not in responses_dict:
            
            composite_image_path = create_composite_image(batch['refer_img'][i], batch['image'][i],\
                output_dir=OUTPUT_DIR, output_image_name=composite_image_name)

            encoded_image = encode_image(composite_image_path)

            # make the prompt
            response = make_request(encoded_image, exemplar_images=exemplar_images, exemplar_captions=exemplar_captions, max_examples=10)
            api_requests += 1

            if PRINT_VERBOSE_RESPONSE:
                rate_limit_info = {
                    'limit_requests': response.headers.get('x-ratelimit-limit-requests'),
                    'limit_tokens': response.headers.get('x-ratelimit-limit-tokens'),
                    'remaining_requests': response.headers.get('x-ratelimit-remaining-requests'),
                    'remaining_tokens': response.headers.get('x-ratelimit-remaining-tokens'),
                    'reset_requests': response.headers.get('x-ratelimit-reset-requests'),
                    'reset_tokens': response.headers.get('x-ratelimit-reset-tokens')
                }
                print("Rate Limit Info:", rate_limit_info)

                response_json = response.json()
                try:
                    print(response_json)
                except UnicodeEncodeError:
                    print("Skip unicode error")              

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                responses_dict[composite_image_name] = content

            # Save after each successful caption to avoid losing progress
            with open(OUTPUT_JSON, 'w') as json_file:
                json.dump(responses_dict, json_file, indent=4)

            num_examples += 1
            #if num_examples > 5: 

            if num_examples >= NUM_EXAMPLES +1:
                return num_examples, api_requests  # Exit the loop after reaching the desired number of examples
    
    return num_examples, api_requests





with open(json_path, 'r') as json_file:
    captions_dict = json.load(json_file)


# Encode images to base64
encoded_exemplar_images = []
for image_name in captions_dict:
    image_path = os.path.join(captions_dir, image_name)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_exemplar_images.append(encoded_image)

# Prepare the shot images and captions for the request
exemplar_images = encoded_exemplar_images
exemplar_captions = list(captions_dict.values())


# Load existing annotations if available

if USE_EXISTING_CAPTIONS and os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, 'r') as json_file:
        responses_dict = json.load(json_file)
else:
    responses_dict = {}

# Calculate the total number of items if known, e.g., len(dataset)
# For demonstration, let's assume an unknown total size initially
total_items = len(dataloader.dataset)  # If you know the total dataset size

# Initialize tqdm with the total number of items
pbar = tqdm(total=total_items, desc="Processing items")

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# Counter for the number of examples
num_examples = 0
api_requests = 0


# Iteration over the dataloader
for batch in dataloader:
    # Assuming each batch is of size 4 based on the DataLoader setup
    encoded_images_batch = []
    for i in range(len(batch['image'])):  # Assuming batch['image'] exists and contains the images
        composite_image_path = create_composite_image(batch['refer_img'][i], batch['image'][i],
                                                      output_dir=OUTPUT_DIR,
                                                      output_image_name=f"composite_{num_examples + i}.jpg")
        encoded_image = encode_image(composite_image_path)
        encoded_images_batch.append(encoded_image)

    # Process the entire batch of 4 images together
    if encoded_images_batch:
        process_batch(encoded_images_batch, exemplar_images, exemplar_captions, num_examples, api_requests, is_training=False)
        num_examples += len(encoded_images_batch)  # Update the number of examples processed
        api_requests += 1  # Assuming one request per batch processed

    # Check if the number of processed examples has reached the desired count
    if num_examples >= NUM_EXAMPLES:
        print(f"Processed {num_examples} examples.")
        break  # Exit the loop if desired number of examples are processed

pbar.close()