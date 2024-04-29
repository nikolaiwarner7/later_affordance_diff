import os
import sys
sys.path.append("/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/")
sys.path.append("/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/")
import base64
import requests
from PIL import Image, ImageOps
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
import random


# Access environment variables
SPLIT = os.getenv('SPLIT', 'train')  # Default to 'train' if not set
NUM_EXAMPLES = int(os.getenv('NUM_EXAMPLES', 10))  # Default to 20 if not set
NUM_SHOTS = int(os.getenv('NUM_SHOTS', 20))  # Default to 20 if not set

print(SPLIT, NUM_EXAMPLES, NUM_SHOTS)

DATASET_NAME = 'Kinetics'
#DATASET_NAME = random.choice(['Kinetics', 'HVU'])
# Choose which one

OUTPUT_PATH = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/"

# the config file including information about the data module
config_filename = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/configs/latent-diffusion/affordance.yaml"

# switch to api key if count exceeds certain limit

USE_EXISTING_CAPTIONS = True
# Rate limits are 500 + now. Request increase on acct.
PRINT_VERBOSE_RESPONSE = False

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

    #json_path = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/example_captions_for_gpt/example_captions_v2_4_1.json"
    #captions_dir = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/example_captions_for_gpt/"

elif DATASET_NAME == 'HVU':  # HVU
    OUTPUT_DIR = os.path.join(OUTPUT_PATH,'kulal_hvu_composites_%s/' % SPLIT)
    OUTPUT_JSON = os.path.join(OUTPUT_PATH, 'kulal_hvu_composites_%s.json' % SPLIT)
    # Redo these later
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

# Setup Train DataLoader and Validation DataLoader 
# based on train/val SPLIT


if SPLIT=='train':
    dataloader = DataLoader(data.datasets["train"], batch_size=data.batch_size,
                              num_workers=data.num_workers, shuffle=True)
elif SPLIT == 'val':
    dataloader = DataLoader(data.datasets["validation"], batch_size=data.batch_size,
                            num_workers=data.num_workers, shuffle=False)
else:
    print("Error loading dataloader, check SPLIT.")


# Function to create and save composite image (same as in REF script)
# Assuming these are always 256,256 squares which they are from preproc operations.
def create_composite_image(image1_tensor, image2_tensor, output_dir, output_image_name):
    # Convert tensors to PIL Images, adjusting image2_tensor as described
    to_pil_image = ToPILImage()
    image1 = to_pil_image(image1_tensor.cpu().permute(2, 0, 1))
    image2 = to_pil_image(0.5 * (image2_tensor.cpu().permute(2, 0, 1) + 1))
    
    # Resize images to have the height of 256 pixels, maintaining aspect ratio
    target_height = 256
    image1 = image1.resize((256, target_height), Image.ANTIALIAS)
    image2 = image2.resize((256, target_height), Image.ANTIALIAS)
    
    # Create a new image with the desired output dimensions (512x256 pixels)
    composite_image = Image.new('RGB', (512, 256))
    
    # Paste the images side by side in the new image
    composite_image.paste(image1, (0, 0))
    composite_image.paste(image2, (256, 0))  # Paste the second image at the 256-pixel mark
    
    # Construct the output image path and save the composite image
    output_image_path = os.path.join(output_dir, output_image_name)
    composite_image.save(output_image_path)
    
    return output_image_path

# Function to make a request to OpenAI API (same as in REF script)
def make_request(encoded_image, shot_images=None, shot_captions=None, max_examples=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = []

    # Determine the number of examples to use based on max_examples
    if shot_images and shot_captions and len(shot_images) == len(shot_captions):
        num_examples = len(shot_images) if max_examples is None else min(max_examples, len(shot_images))

        messages.append({
            "role": "system",
            "content": "I will provide examples of scene difference captions given composite images (of 2 different scenes from the same video)\
                    There are two frames in the image. Caption the difference between frames (assume left or top image comes first)."
        })        

        for i in range(num_examples):
            detail_level = random.choice(["low", "high"])  # Randomly select 'low' or 'high' detail level
            #detail_level = "low"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shot_images[i]}", "detail": detail_level}},
                    {"type": "text", "text": shot_captions[i]}
                ]
            })

    # Add the actual request for the target image
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
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_batch(batch, encoded_images, shot_captions, num_examples, api_requests, is_training):
    global iteration_count
    global api_key

    batch_size = len(batch['data_idx'])

    #ipdb.set_trace()

    # process the batch
    for i in range(batch_size):  # Use the actual batch size
        composite_image_name = "composite_%s.jpg" % batch['data_idx'][i]
        if composite_image_name not in responses_dict:
            composite_image_path = create_composite_image(batch['refer_img'][i], batch['image'][i],
                                                          output_dir=OUTPUT_DIR, output_image_name=composite_image_name)
            encoded_image = encode_image(composite_image_path)

            # make the prompt
            response = make_request(encoded_image, shot_images=encoded_images, shot_captions=shot_captions, max_examples=NUM_SHOTS)
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
encoded_images = []
for image_name in captions_dict:
    image_path = os.path.join(captions_dir, image_name)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_images.append(encoded_image)

# Prepare the shot images and captions for the request
shot_images = encoded_images
shot_captions = list(captions_dict.values())


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


# Iteration for validation dataloader
for batch in dataloader:

    ipdb.set_trace()
    num_examples, api_requests = process_batch(batch, encoded_images, shot_captions, num_examples, api_requests, is_training=False)

    pbar.update(len(batch))

    # Check if the number of processed examples has reached the desired count
    if num_examples >= NUM_EXAMPLES:
        print(f"Reached the desired number of examples: {NUM_EXAMPLES}")
        break  # Exit the loop

pbar.close()

