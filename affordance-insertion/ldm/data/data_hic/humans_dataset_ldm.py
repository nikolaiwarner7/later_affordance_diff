from __future__ import annotations

__all__ = ["HumansDataset", "HumansDatasetSubset"]

import os
import sys

sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes')
sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/open_pose')
from draw_pose_tensor import draw_pose_tensor
# parent import issue later
#sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/open_pose')
#from draw_pose import draw_pose

import random
import pathlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import ClassVar, Union, Optional
import numpy as np
from einops import rearrange, repeat

from PIL import Image

import torch
from torch.utils.data import ConcatDataset, Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from transformers import CLIPFeatureExtractor
import bisect
from scipy import ndimage

from . import utils
from .database import Database, ImageDatabase
from .augment import AugmentPipe
from .comodgan_mask import RandomMaskRect
import ipdb
import json
import inspect

# Decides what percent of the time to keep frames fixed for captioning,vs shuffling, setting caption to null and learning img affordances.
FIX_FRAMES_FRACTION = float(os.environ.get('FIX_FRAMES_FRACTION', 0.9)) 
print("fix_frames_fraction", FIX_FRAMES_FRACTION)


@dataclass
class HumansDataset(ConcatDataset):
    path: Union[str, os.PathLike[str]]
    resolution: int = 256
    num_frames: int = 1
    spacing: int = 4
    flip_p: float = 0.5
    deterministic: bool = False
    split: Optional[str] = "train"
    num_keypoints: ClassVar[int] = 18
    test_length: ClassVar[int] = 500e3
    seed: ClassVar[int] = 10000
    json_train_path: ClassVar[str] = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kulal_kinetics_composites_train.json"
    json_val_path: ClassVar[str] = "/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/kulal_kinetics_composites_val.json"
 
    # masks are ratios of bbox, mask, scribble, random bbox
    # modes are train, test, overfit, swap, halves
    # train - [25, 25, 25, 25], dil 10
    # test - [50, 50, 0, 0], dil 0
    # overfit - [100, 0, 0, 0], dil 0
    # swap - [50, 50, 0, 0], dil 10
    # halves - [50, 50, 0, 0], dil 0
    config: Dict[str] = field(default_factory=dict)

    def __post_init__(self):

        # Determine which JSON file to load
        # Check on this
        json_path = self.json_train_path if self.config['mode'] == 'train' else self.json_val_path
        # Load the JSON file
        with open(json_path, 'r') as json_file:
            self.responses_dict = json.load(json_file)


        print(self.config)
        assert self.config is not None
        assert self.split in (None, "train", "test")

        path = pathlib.Path(self.path)
        datasets = []
        self.subsets = []

        
        for subset_path in path.iterdir():
            self.subsets.append(subset_path.name)

            dataset = HumansDatasetSubset(
                subset_path,
                self.resolution,
                self.num_frames,
                self.spacing,
                self.deterministic,
                self.flip_p,
                self.config,
                responses_dict=self.responses_dict,  # Pass the dictionary here
            )
            datasets.append(dataset)
        #
        self.subsets.sort()
        super().__init__(datasets)

        length = self.cumulative_sizes[-1]
        generator = torch.Generator().manual_seed(self.seed)
        self.indices = torch.randperm(length, generator=generator).tolist()
        
        # Not relevant to our data
        # if self.split == "test":
        #     self.indices = self.indices[: self.test_length]
        # elif self.split == "train":
        #     self.indices = self.indices[self.test_length :]

        if self.config['mode'] == 'overfit':
            self.indices = self.indices[:1]


    def __getitem__(self, index: int):
        if self.config['mode'] == 'overfit':
            return super().__getitem__(self.indices[index // 1000000])
        else:
            return super().__getitem__(self.indices[index])

    def __len__(self) -> int:
        if self.config['mode'] == 'overfit':
            return len(self.indices) * 1000000
        else:
            return len(self.indices)


@dataclass
class HumansDatasetSubset(Dataset):
    path: Union[str, os.PathLike[str]]
    resolution: int = 128
    num_frames: int = 1
    spacing: int = 4
    deterministic: bool = False
    num_keypoints: ClassVar[int] = 18
    flip_p: float = 0.5
    config: Dict[str] = field(default_factory=dict)
    responses_dict: dict = field(default_factory=dict)  # Add this line

    def __post_init__(self):
        # self.flip = transforms.RandomHorizontalFlip(p=self.flip_p)
        path = pathlib.Path(self.path)
        #
        #
        #ipdb.set_trace()
        frames_db_path = str(path.joinpath("frames_db"))
        print(f'loading {frames_db_path}')
        self.frames_db = ImageDatabase(
            frames_db_path, readahead=False, lock=False
        )

        #ipdb.set_trace()

        # 4_14 add back poses_db_path from hallucinating_scenes repository
        #ipdb.set_trace()
        new_poses_path = '/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/4_25_pose_results.json'

        # Load the JSON data from the file
        with open(new_poses_path, 'r') as file:
            new_pose_data = json.load(file)
        self.new_pose_data = new_pose_data

        poses_db_path = str(path.joinpath("poses_db"))
        self.poses_db = Database(poses_db_path, readahead=False, lock=False)

        # Initialize train poses_DB, that gets set regardless of train/val
        # contains dropout conditions for pose
        TRAIN_PATH =  '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/training_data/kinetics'  # to validation data

        #ipdb.set_trace()
        dropout_poses_db_path = Path(TRAIN_PATH)
        dropout_poses_db_path = str(dropout_poses_db_path.joinpath("poses_db"))
        self.dropout_poses_db = Database(dropout_poses_db_path, readahead=False, lock=False)
    
        #ipdb.set_trace()
    
        clips_db_path = str(path.joinpath("clipmask_db"))
        clips_db = Database(clips_db_path, lock=False)
        
        boxes_db_path = str(path.joinpath("masks_db"))
        self.boxes_db = Database(boxes_db_path, readahead=False, lock=False)

        #ipdb.set_trace()
        # Load the JSON file
        with open('train_clip_to_caption.json', 'r') as file:
            self.clip_to_caption_mapping = json.load(file)

        # Load the JSON file
        with open('train_clip_to_frame_indices.json', 'r') as file:
            self.clip_to_frame_mapping = json.load(file)

        self.keys = []
        self.min_length = self.spacing * (self.num_frames - 1) + 1

        #ipdb.set_trace()
        clip_keys = clips_db.keys()
        for clip_key in clip_keys:
            frame_keys = clips_db[clip_key]
            if len(frame_keys) >= 2:
                self.keys.append(frame_keys)

        if self.deterministic:
            self.seed = int.from_bytes(path.name.encode(), byteorder="big")
            self.seed &= 0xFFFFFFFF
        else:
            self.seed = None

        

        if 'clip' in self.config:
            print(f"Using CLIP {self.config['clip']}")
            self.feat_extract = CLIPFeatureExtractor.from_pretrained(self.config['clip'])
        else:
            print(f"Using CLIP openai/clip-vit-large-patch14")
            self.feat_extract = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.first_aug_pipe = AugmentPipe(\
            xflip=0, rotate90=0, xint=0, xint_max=0.125,
            scale=0.0, rotate=0.0, aniso=0.0, xfrac=0, scale_std=0.2, rotate_max=0.2, aniso_std=0.2, xfrac_std=0.125,
            brightness=0.2, contrast=0.2, lumaflip=0, hue=0, saturation=0.2, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=0.5,
            imgfilter=0.1, imgfilter_bands=[1,1,1,1], imgfilter_std=0.2,
            noise=0.1, cutout=0.0, noise_std=0.05, cutout_size=0.5,
        )
        self.second_aug_pipe = AugmentPipe(\
            xflip=0, rotate90=0, xint=0, xint_max=0.125,
            scale=0.4, rotate=0.4, aniso=0.2, xfrac=0, scale_std=0.2, rotate_max=0.2, aniso_std=0.2, xfrac_std=0.125,
            brightness=0.0, contrast=0.0, lumaflip=0, hue=0, saturation=0.0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=0.5,
            imgfilter=0.0, imgfilter_bands=[1,1,1,1], imgfilter_std=0.2,
            noise=0.0, cutout=0.2, noise_std=0.05, cutout_size=0.5,
        )
        
        self.last_valid_sample = None

    def _process_person(self, frame, mask, bbox, center):
        
        # mask_01 3D also 0-1 to extract person
        mask_01 = mask[:, :, np.newaxis].astype(np.float32)
        mask_01 = np.repeat(mask_01, 3, axis=2)
        frame = np.asarray(frame).astype(np.float32)

        # data augmentation for refer person
        aug_frame, aug_mask = torch.tensor(frame), torch.tensor(mask_01)
        aug_frame = rearrange(aug_frame, 'h w c -> 1 c h w')
        aug_frame = 2 * aug_frame / 255.0 - 1
        aug_mask = rearrange(aug_mask, 'h w c -> 1 c h w')
        if self.config['mode'] == 'train' and self.config['augment']:
            aug_frame = self.first_aug_pipe(aug_frame)
        aug_frame = aug_frame * aug_mask
        shift_x = 128 - center[0]
        shift_y = 128 - center[1]
        aug_frame = torchvision.transforms.functional.affine(aug_frame, 
                        translate=[int(shift_x), int(shift_y)],
                        angle=0, scale=1, shear=0, fill=0.0)
        if self.config['mode'] == 'train' and self.config['augment']:
            aug_frame = self.second_aug_pipe(aug_frame)
        aug_frame = (aug_frame + 1) * 255.0 / 2
        aug_frame = torch.clamp(aug_frame, min=0.0, max=255.0)
        aug_frame = rearrange(aug_frame, '1 c h w -> h w c')
        person_rgb = aug_frame.numpy()

        return person_rgb

    def _process_frame(self, frame, mask, bbox, center):
        
        # mask_01 3D also 0-1 to extract person
        mask_01 = mask[:, :, np.newaxis].astype(np.float32)
        mask_01 = np.repeat(mask_01, 3, axis=2)
        frame = np.asarray(frame).astype(np.float32)

        mask_type = random.choices(\
            ['bbox', 'pmask', 'scribble', 'smallbox', 'largebox'], \
            weights=self.config['masks'], k=1)[0]

        if mask_type == 'bbox':
            mask_rgb = mask * 0
            if self.config['dilation_mode'] == 'random':
                dilation = random.randint(0, self.config['dilation'])
            elif self.config['dilation_mode'] == 'fixed':
                dilation = self.config['dilation']
            else:
                raise ValueError
            y_min = max(int(bbox[1] - dilation // 2), 0)
            y_max = max(int(bbox[3] + (dilation + 1) // 2), 0)
            x_min = max(int(bbox[0] - dilation // 2), 0)
            x_max = max(int(bbox[2] + (dilation + 1) // 2), 0)
            if self.config['mode'] == 'halves':
                if random.random() < 0.5:
                    y_min = (y_min + y_max) // 2
                else:
                    y_max = (y_min + y_max) // 2
            mask_rgb[y_min:y_max, x_min:x_max] = 1
        elif mask_type == 'smallbox':
            y_min, y_max, x_min, x_max = int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])
            area = (x_max - x_min) * (y_max - y_min)
            percent = random.uniform(25, 100)
            area = area * percent / 100
            sq = np.sqrt(area)
            width_new = random.uniform(min(sq, (x_max - x_min)), (x_max - x_min))
            height_new = min(area / width_new, (y_max - y_min))
            x_min_new = random.randint(x_min, x_max - int(width_new))
            y_min_new = random.randint(y_min, y_max - int(height_new))
            mask_rgb = mask * 0
            mask_rgb[y_min_new : y_min_new + int(height_new), \
                x_min_new : x_min_new + int(width_new)] = 1
        elif mask_type == 'pmask':
            mask_rgb = mask
            struct = ndimage.generate_binary_structure(2, 1)
            if self.config['dilation_mode'] == 'random':
                dilation = random.randint(0, self.config['dilation'])
            elif self.config['dilation_mode'] == 'fixed':
                dilation = self.config['dilation']
            else:
                raise ValueError
            mask_rgb = ndimage.binary_dilation(mask_rgb, structure=struct, iterations=dilation)
            mask_rgb = mask_rgb.astype(np.uint8)
        elif mask_type == 'scribble':
            mask_rgb = mask * 0
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            size = max(y_max - y_min, x_max - x_min)
            rand_submask = RandomMaskRect(y_max - y_min, x_max - x_min, hole_range=[0.2, 0.9])
            mask_rgb[y_min:y_max, x_min:x_max] = 1 - rand_submask
        elif mask_type == 'largebox':
            mask_rgb = mask * 0
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            size = max(y_max - y_min, x_max - x_min) // 2

            ratio = random.randint(5, 40)
            y_min_len = size * (100 + ratio) / 100.0
            ratio = random.randint(5, 40)
            y_max_len = size * (100 + ratio) / 100.0
            ratio = random.randint(5, 40)
            x_min_len = size * (100 + ratio) / 100.0
            ratio = random.randint(5, 40)
            x_max_len = size * (100 + ratio) / 100.0

            y_min = max(0, int(center[1] - y_min_len))
            y_max = max(0, int(center[1] + y_max_len))
            x_min = max(0, int(center[0] - x_min_len))
            x_max = max(0, int(center[0] + x_max_len))
            mask_rgb[y_min:y_max, x_min:x_max] = 1
            
        mask_01 = mask_rgb[:, :, np.newaxis].astype(np.float32)
        mask_01 = np.repeat(mask_01, 3, axis=2)
        #
        masked_frame = frame * (1 - mask_01) + np.full_like(frame, 127.5) * (mask_01)
        mask_rgb = mask_rgb * 255

        return frame, masked_frame, mask_rgb

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Check if the environment variable is set and convert it to a boolean
        distance_check_mode = os.getenv('DISTANCE_CHECK_MODE', 'None')

        frame_keys = self.keys[index]
        video_idx = int(frame_keys[0].split("_")[1]) # dataloader stores index of sparse videos, not the video number
        

        # 3_28 test
        #if self.config['mode'] == 'test': ipdb.set_trace()\

        # This condition here, decides 20% of the time, to drop the fixed frames and caption for training, so it trains on
        # all available frame permutations.
        fix_frames_decision = random.random() <=  FIX_FRAMES_FRACTION # True 80% of the time

        if self.config['mode'] == 'test': # dont think val needs to fix but check
            caption_index = 'composite_%s.jpg' % str(index) # check this
            scene_difference_caption = self.responses_dict[caption_index]

        elif self.config['mode'] == 'train':
            if str(video_idx) in self.clip_to_caption_mapping.keys(): 
                caption_index = self.clip_to_caption_mapping[str(video_idx)]
                caption_full_indx = 'composite_%s.jpg' % caption_index
                if caption_full_indx in self.responses_dict.keys():
                    scene_difference_caption = self.responses_dict[caption_full_indx]
                else:
                    #print("Bad key")
                    scene_difference_caption = '' # Null
            else: 
                scene_difference_caption = ''


        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)
        else:
            self.generator = None

        
        index_start = torch.randint(
            0, len(frame_keys), (), generator=self.generator
        ).item()
        if self.config['mode'] == 'test' or self.config['mode'] == 'swap':
            index_start = 0

        if self.config['mode'] == 'train' and str(video_idx) in self.clip_to_caption_mapping.keys(): # fix the index_refer and index_start
            if fix_frames_decision:  # 80%: Fix the frames based on the mapping
                index_start = self.clip_to_frame_mapping[str(video_idx)][1]  # target is 2nd entry
            else:  # 20%: Choose frames randomly and drop the caption
                scene_difference_caption = ''
                # Ensure index_start is chosen randomly (the logic to do so should already exist outside this block)

        if index_start >= len(frame_keys) or index_start < 0:
            # about 15 bad examples from 4_13 procedure, less than 0.1% so just drop 
            index_start = 0
        frame_key = frame_keys[index_start]
        frame = self.frames_db[frame_key]

        ipdb.set_trace()

        # Extract keypoints and scores
        keypoints = self.new_pose_data[frame_key]['keypoints']
        scores = self.new_pose_data[frame_key]['keypoints_score']
        combined_keypoints = [(x, y, score) for (x, y), score in zip(keypoints, scores)]
        # Pad keypoints to ensure there are 18 keypoints to match old OpenPose format
        if len(combined_keypoints) < 18:
            combined_keypoints += [(0, 0, 0)] * (18 - len(combined_keypoints))
        # Convert to a tensor and reshape
        frame_pose_target = torch.tensor(combined_keypoints, dtype=torch.float64).unsqueeze(0)  # Adds the batch dimension

        """
        if frame_key in self.poses_db:
            frame_pose_target = self.poses_db[frame_key]
        else:

            prefix = frame_key.split("_frame")[0]
            matching_frames = [frame for frame in self.poses_db.keys() if frame.startswith(prefix)]
            if matching_frames:
                frame_pose_target = self.poses_db[matching_frames[-1]]
                #print("substituted target pose from", str(len(matching_frames)))
            else:
                #print("zeroed target pose not in db")
                frame_pose_target = torch.zeros((1,18,3))
        """

        assert frame.height == frame.width
    
        refer_frame_keys = frame_keys.copy()
        refer_frame_keys.remove(frame_key)
        index_refer = torch.randint(
            0, len(refer_frame_keys), (), generator=self.generator
        ).item()
        index_refer = frame_keys.index(refer_frame_keys[index_refer])


        if self.config['mode'] == 'train' and str(video_idx) in self.clip_to_caption_mapping.keys(): # fix the index_refer and index_start
            if fix_frames_decision:  # 80%: Fix the frames based on the mapping
                index_refer = self.clip_to_frame_mapping[str(video_idx)][0]  # ref is first entry
            else:  # 20%: Choose frames randomly and drop the caption
                scene_difference_caption = ''
                # Ensure index_refer is chosen randomly (assuming the logic to do so should already exist outside this block)


        if index_refer >= len(frame_keys) or index_refer < 0:
            # about 15 bad examples from 4_13 procedure, less than 0.1% so just drop 
            index_refer = len(frame_keys) - 1

        # get main frame and refer frame info
        if self.config['mode'] == 'swap':
            index_refer = 0
        if self.config['mode'] == 'test':
            index_refer = len(frame_keys) - 1
        refer_img = self.frames_db[frame_keys[index_refer]]
        refer_dict = self.boxes_db[frame_keys[index_refer]]


        # Extract keypoints and scores
        keypoints = self.new_pose_data[frame_keys[index_refer]]['keypoints']
        scores = self.new_pose_data[frame_keys[index_refer]]['keypoints_score']
        combined_keypoints = [(x, y, score) for (x, y), score in zip(keypoints, scores)]
        # Pad keypoints to ensure there are 18 keypoints to match old OpenPose format
        if len(combined_keypoints) < 18:
            combined_keypoints += [(0, 0, 0)] * (18 - len(combined_keypoints))
        # Convert to a tensor and reshape
        frame_pose_refer = torch.tensor(combined_keypoints, dtype=torch.float64).unsqueeze(0)  # Adds the batch dimension

        """ Old before 4_28
        if frame_keys[index_refer] in self.poses_db:
            frame_pose_refer = self.poses_db[frame_keys[index_refer]]
        else:
            #ipdb.set_trace()
            prefix = frame_keys[index_refer].split("_frame")[0]
            matching_frames = [frame for frame in self.poses_db.keys() if frame.startswith(prefix)]
            if matching_frames:
                #if self.config['mode'] == 'train': ipdb.set_trace()
                frame_pose_refer = self.poses_db[matching_frames[-1]]
                #ipdb.set_trace()
                #print("substituted refer pose from", str(len(matching_frames)))
            else:
                #print("zeroed refer pose not in db")
                frame_pose_refer = torch.zeros((1,18,3))
        """


        # Depending on the mode set by the environment variable, apply different checks:
        if distance_check_mode == 'check_average':
            MAX_AVG_DIST = 40
            MIN_AVG_DIST = 10
            if frame_pose_target.numel() > 0 and frame_pose_refer.numel() > 0:
                keypoints_target = frame_pose_target[0, :, :2]
                keypoints_refer = frame_pose_refer[0, :, :2]
                valid_keypoints_mask = (keypoints_target.sum(dim=1) != 0) & (keypoints_refer.sum(dim=1) != 0)
                if valid_keypoints_mask.sum() > 0:
                    distances = torch.norm(keypoints_target[valid_keypoints_mask] - keypoints_refer[valid_keypoints_mask], dim=1)
                    average_distance = distances.mean()
                    if average_distance > MAX_AVG_DIST or average_distance < MIN_AVG_DIST:
                        return self.last_valid_sample  # Filter out this example

        elif distance_check_mode == 'check_hip':
            MAX_HIP_DIST = 40
            MIN_HIP_DIST = 10
            if frame_pose_target.numel() > 0 and frame_pose_refer.numel() > 0:
                # Keypoints 11 and 12 are hips, index 10 and 11 in zero-based indexing
                #ipdb.set_trace()
                hips_target = frame_pose_target[0, [10, 11], :2]
                hips_refer = frame_pose_refer[0, [10, 11], :2]
                if (hips_target.sum(dim=1) != 0).all() and (hips_refer.sum(dim=1) != 0).all():
                    hip_distance = torch.norm(hips_target - hips_refer, dim=1).mean()
                    if hip_distance > MAX_HIP_DIST or hip_distance < MIN_HIP_DIST:
                        return self.last_valid_sample  # Filter out this example based on hip translation




        if self.config['mode'] == 'swap':    
            # refer_clip_idx = random.randint(0, len(self.keys) - 1)
            refer_clip_idx = len(self.keys) - 1 - index
            refer_img = self.frames_db[self.keys[refer_clip_idx][index_refer]]
            refer_dict = self.boxes_db[self.keys[refer_clip_idx][index_refer]]
        frame_dict = self.boxes_db[frame_key]
        
        


        refer_center = refer_dict['center']
        refer_mask = refer_dict['mask']
        refer_mask = np.unpackbits(refer_mask.reshape(256, -1), axis=-1)
        frame_center = frame_dict['center']
        frame_mask = frame_dict['mask']
        frame_mask = np.unpackbits(frame_mask.reshape(256, -1), axis=-1)
                      
        frame_bbox = frame_dict['box']
        refer_bbox = refer_dict['box']
        is_uncond = 0  

        if self.config['data_type'] == 'image':
            refer_person = self._process_person(frame, frame_mask, frame_bbox, frame_center)  
            frame, frame_masked, frame_mask = self._process_frame(frame, frame_mask, frame_bbox, frame_center)
        else:    
            frame, frame_masked, frame_mask = self._process_frame(frame, frame_mask, frame_bbox, frame_center)
            refer_person = self._process_person(refer_img, refer_mask, refer_bbox, refer_center)

        
        refer_img_tensor = torchvision.transforms.ToTensor()(refer_img).permute(1, 2, 0)
    
        ### FIXME only if support for res needs to be added
        # if frame.height != self.resolution:
        #     pose[:, :2] *= self.resolution / frame.height

        #     size = (self.resolution, self.resolution)
        #     frame = frame.resize(size, resample=Image.LANCZOS)
        #     frame_masked = frame_masked.resize(size, resample=Image.LANCZOS)
        #     frame_mask = frame_mask.resize(size, resample=Image.LANCZOS)
        #     refer_person = refer_person.resize(size, resample=Image.LANCZOS)

        if random.random() < self.flip_p:            
            frame, frame_masked, frame_mask, refer_person = \
                np.fliplr(frame), np.fliplr(frame_masked), np.fliplr(frame_mask), \
                np.fliplr(refer_person)

        def _n(frame, mode=None):
            frame = np.array(frame).astype(np.float32)
            if mode != 'mask':
                frame = 2.0 * frame / 255.0 - 1.0
            else:
                frame = frame / 255.0
            return frame

        zero_person = np.full_like(np.array(refer_person), 127.5)
        zero_encoding = self.feat_extract(Image.fromarray(zero_person.astype(np.uint8)), return_tensors="pt")

        # First image dropout, independent of other types
        drop_type = random.random()


        if (drop_type <= 0.10 and self.config['mode'] == 'train'):
            refer_person = np.full_like(np.array(refer_person), 127.5)
            frame_masked = np.full_like(np.array(frame_masked), 127.5)
            frame_mask = np.full_like(np.array(frame_mask), 255)
            batch_encoding = self.feat_extract(Image.fromarray(refer_person.astype(np.uint8)), return_tensors="pt")


            is_uncond = 1

        elif (drop_type > 0.10 and drop_type <= 0.20 and self.config['mode'] == 'train') or self.config['zero_person']:
            refer_person = np.full_like(np.array(refer_person), 127.5)
            batch_encoding = self.feat_extract(Image.fromarray(refer_person.astype(np.uint8)), return_tensors="pt")

            is_uncond = 1
        else:
            batch_encoding = self.feat_extract(Image.fromarray(refer_person.astype(np.uint8)), return_tensors="pt")

        # Independent dropout for all 3 channels
        drop_caption = random.random() < 0.2  # 20% chance to drop the caption
        drop_poses = random.random() < 0.2    # 20% chance to drop both poses together


        ipdb.set_trace()
        if drop_caption:
            scene_difference_caption = ""  # Drop the caption

        if drop_poses:
            # Drop the poses
            prefix = 'clip_0006465'
            matching_frames = [frame for frame in self.dropout_poses_db.keys() if frame.startswith(prefix)]
            if matching_frames:
                frame_pose_target = self.dropout_poses_db[matching_frames[0]]
                frame_pose_refer = self.dropout_poses_db[matching_frames[-1]]

        #if drop_poses: ipdb.set_trace()

        frame, frame_masked, frame_mask, refer_person, zero_person = \
                _n(frame), _n(frame_masked), _n(frame_mask, mode='mask'), \
                _n(refer_person), _n(zero_person)      
    

        #if self.config['mode'] == 'test': ipdb.set_trace()\


        # process frame_pose_refer into the 2D pose cond shape
        test1 = rearrange(batch_encoding['pixel_values'][0], 'c h w -> h w c')
        black_img_for_pose = torch.zeros(test1.shape)
        target_pose_vis = draw_pose_tensor(black_img_for_pose, frame_pose_target).numpy().astype('uint8')
        refer_pose_vis = draw_pose_tensor(black_img_for_pose, frame_pose_refer).numpy().astype('uint8')

        #ipdb.set_trace()
        """  for testing/ visualizing
        Image.fromarray(target_pose_vis.numpy().astype('uint8')).save('4_22_test_pose_target_vis.png')
        Image.fromarray(refer_pose_vis.numpy().astype('uint8')).save('4_22_test_pose_refer_vis.png')
        Image.fromarray(((refer_person + 1) * 127.5).astype(np.uint8)).save('4_22_test_refer_img.png')
        bbox = [69.427345, 23.686796, 180.64845, 254.30637]  # Define the bounding box
        Image.open('4_22_test_pose_refer_vis.png').crop(bbox).save('cropped_4_22_test_pose_refer_vis.png')
        Image.fromarray(((frame + 1) * 127.5).astype(np.uint8)).save('4_22_test_target_img.png')
        ipdb.set_trace()
        """
        # 4_17 disabling img encoding of pose vis for now.

        example = {'image': frame, 'masked_image': frame_masked, 'mask': frame_mask, \
            'refer_person_clip': rearrange(batch_encoding['pixel_values'][0], 'c h w -> h w c'), \
            'refer_person': refer_person, \
            'zero_person_clip': rearrange(zero_encoding['pixel_values'][0], 'c h w -> h w c'), \
            'zero_person': zero_person, \
            'uncond_mask': is_uncond, \
            'scene_difference_caption': scene_difference_caption, \
            'refer_img': refer_img_tensor,
            'data_idx': str(index),
            'frame_pose_target': frame_pose_target,
            'frame_pose_refer' : frame_pose_refer,
            'target_pose_vis': target_pose_vis,
            'refer_pose_vis': refer_pose_vis,
            }
        
        self.last_valid_sample = example
        return example

    def __len__(self):


        return len(self.keys)

