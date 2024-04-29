import argparse, os, sys, glob
sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion')
sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic')

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from database import Database, ImageDatabase
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
import ipdb


from pathlib import Path
from cleanfid import fid
import shutil

import copy
from einops import rearrange, repeat
import scipy.ndimage as ndimage


# Set wandb to offline mode
#os.environ['WANDB_MODE'] = 'offline'

import wandb
wandb.init(project="affordance")
# This run was from the train/val split run.
# Current path

# Check if the environment variable is set and convert it to a boolean
distance_check_mode = os.getenv('DISTANCE_CHECK_MODE', 'None')
print("distance check mode for 8750 (default None):", distance_check_mode)

MODEL_PATH = '/coc/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-04-23T13-04-38_affordance/checkpoints/epoch=000119.ckpt'

DATA_PATH =  '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/validation_data'  # to validation data
INF_MODE = 'test' 


#DATA_PATH =  '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/training_data'  # to training data
#INF_MODE = 'train' # take one frame, place in another
#make sure to set to train for the training data bc then it picks specific frames.

print(DATA_PATH)


CONFIG_PATH = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/configs/latent-diffusion/affordance.yaml'
#CONFIG_PATH = 'models/affordance/config.yaml'
CFG_SCALE = 4.0

#Stable and used to extract unconditional pose representation
TRAIN_PATH =  '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/full_kinetics_data_our_sampling_3_18/training_data/kinetics'  # to validation data


NUM_BATCHES_BREAK = 40
INFERENCE_BATCH_SIZE = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        '--blend', 
        action='store_true',
        help="blend the pred image with masked image"
    )
    opt = parser.parse_args()

    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(MODEL_PATH)["state_dict"], strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    config['data']['params']['batch_size'] = INFERENCE_BATCH_SIZE
    config['data']['params']['validation']['params']['config']['mode'] = INF_MODE
    config['data']['params']['validation']['params']['path'] = DATA_PATH
    data = instantiate_from_config(config.data)
    data.setup()
    dataset = data._val_dataloader()

    with torch.no_grad():
        with model.ema_scope():
            for batch_idx, batch in tqdm(enumerate(dataset)):
                for k in batch:
                    x = batch[k]
                    # x = x[None]
                    if not isinstance(x, list):
                        if len(x.shape) != 1:   
                            if len(x.shape) == 3:
                                x = x[..., None]
                            x = rearrange(x, 'b h w c -> b c h w')
                        # x = torch.from_numpy(x)
                        x = x.to(memory_format=torch.contiguous_format).float().to(device)
                    batch[k] = x
                
                #ipdb.set_trace()
                all_samples = []
                for cfg_scale in [CFG_SCALE ]:
                    ### conditional person insertion sampling
                    for idx in range(1):
                        #ipdb.set_trace()
                        cond_dict_sdm = {'concat': {}, 'crossattn': {}}
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["masked_image"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"])
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["refer_person_clip"])
                        # Zero out text.
                        cond_dict_sdm['crossattn']['refer_person_text'] = torch.zeros_like(model.cond_text_model.encode(batch["scene_difference_caption"])).cuda()
                        # Add pose
                        cond_dict_sdm['crossattn']['frame_pose_target'] = model.cond_pose_model.encode(batch["frame_pose_target"])
                        cond_dict_sdm['crossattn']['frame_pose_refer'] = model.cond_pose_model.encode(batch["frame_pose_refer"])

                        cond_dict_sdm['crossattn']['target_pose_vis'] = model.cond_stage_model.encode(batch["target_pose_vis"])
                        
                        
                        cond_dict_sdm['uncond_mask'] = batch["uncond_mask"]
                        
                        #ipdb.set_trace()

                        c_sdm = copy.deepcopy(cond_dict_sdm)
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["zero_person"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"] * 0 + 1)
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["zero_person_clip"])
                        cond_dict_sdm['crossattn']['refer_person_text'] = torch.zeros_like(model.cond_text_model.encode(['']*INFERENCE_BATCH_SIZE)) # null captions

 
                        # 4_14 add back poses_db_path from hallucinating_scenes repository        
                        train_db_path = Path(TRAIN_PATH)
                        poses_db_path = str(train_db_path.joinpath("poses_db"))
                        poses_db = Database(poses_db_path, readahead=False, lock=False)
                        prefix = 'clip_0006465'
                        matching_frames = [frame for frame in poses_db.keys() if frame.startswith(prefix)]
                        if matching_frames:
                            uncond_frame_pose_target = poses_db[matching_frames[0]].cuda().float()
                            uncond_frame_pose_refer = poses_db[matching_frames[-1]].cuda().float()

                        #ipdb.set_trace()

                        # 
                        #uncond_frame_pose_target = torch.zeros((INFERENCE_BATCH_SIZE,18,3)).cuda()
                        cond_dict_sdm['crossattn']['frame_pose_target'] = model.cond_pose_model.encode(uncond_frame_pose_target).repeat(INFERENCE_BATCH_SIZE, 1, 1)
                        cond_dict_sdm['crossattn']['frame_pose_refer'] = model.cond_pose_model.encode(uncond_frame_pose_refer).repeat(INFERENCE_BATCH_SIZE, 1, 1)

                        cond_dict_sdm['crossattn']['target_pose_vis'] = model.cond_stage_model.encode(batch["target_pose_vis"])


                        cond_dict_sdm['uncond_mask'] = batch['uncond_mask'] * 0 + 1
                        uc_sdm = copy.deepcopy(cond_dict_sdm)

                        shape_sdm = (4,)+c_sdm['concat']['masked_image'].shape[2:]             

                        c_cond = c_sdm
                        uc_cond = uc_sdm
                        batch_size = c_sdm['concat']['masked_image'].shape[0]                        
                        shape = shape_sdm
                        samples_ddim, intermediates = sampler.sample(S=opt.steps,       
                                                        conditioning=c_cond,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        unconditional_guidance_scale=cfg_scale,
                                                        unconditional_conditioning=uc_cond,
                                                        verbose=False)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        all_samples.append(x_samples_ddim)         

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                masked_image = torch.clamp((batch["masked_image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                refer_person = torch.clamp((batch["refer_person"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp(batch["mask"], min=0.0, max=1.0).cpu().numpy().transpose(0,2,3,1)

                def _unnorm(x):
                    return torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)

                cfg_predicted_images = [_unnorm(x) for x in all_samples]

                batch_size = batch['image'].shape[0]

                """
                for elem_idx in range(batch_size):
                    # inpainted = predicted_image[elem_idx]
                    # if opt.blend:
                    #     inpainted = (1-mask[elem_idx])*image[elem_idx]+mask[elem_idx]*predicted_image[elem_idx]
                    # inpainted = inpainted.cpu().numpy().transpose(1,2,0)*255
                    # inpainted = Image.fromarray(inpainted.astype(np.uint8))
                    def proc(img):
                        img = img.cpu().numpy().transpose(1,2,0)*255
                        return Image.fromarray(img.astype(np.uint8))

                    all_imgs = [proc(masked_image[elem_idx]), \
                        proc(refer_person[elem_idx]), \
                        proc(image[elem_idx])]

                    for idx in range(len(cfg_predicted_images)):
                        temp_img = proc(cfg_predicted_images[idx][elem_idx])
                        all_imgs.append(temp_img)

                    wandb.log({"cond_inpainting": [wandb.Image(x) for x in all_imgs]}, \
                        step=batch_idx * batch_size + elem_idx)
                """

                for elem_idx in range(batch_size):
                    # Process images as before
                    def proc(img):
                        img = img.cpu().numpy().transpose(1, 2, 0) * 255
                        return Image.fromarray(img.astype(np.uint8))

                    all_imgs_and_captions = [
                        (proc(masked_image[elem_idx]), "Masked Image"),
                        (proc(refer_person[elem_idx]), "Referenced Person"),
                        (proc(image[elem_idx]), "Original Image")
                    ]

                    # Add generated images with captions
                    for idx, cfg_pred_img in enumerate(cfg_predicted_images):
                        temp_img = proc(cfg_pred_img[elem_idx])
                        #ipdb.set_trace()
                        caption = batch["scene_difference_caption"][elem_idx] if batch["scene_difference_caption"] else "Generated Image"
                        all_imgs_and_captions.append((temp_img, caption))

                    # Add the frame_pose_target image # only change
                    target_pose_img = proc(batch["target_pose_vis"][elem_idx])
                    all_imgs_and_captions.append((target_pose_img, "Target Pose Vis"))


                    # Log each image with its caption
                    wandb_images = [wandb.Image(img, caption=caption) for img, caption in all_imgs_and_captions]
                    wandb.log({"cond_inpainting": wandb_images}, step=batch_idx * batch_size + elem_idx)


                if batch_idx > NUM_BATCHES_BREAK:
                    break
                

wandb.finish()
