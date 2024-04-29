3_25 Scripts from Kulal Repository
These are post-preprocessing, used for training
For preprocessing, see: 

Various breakpoints, kept for specific debugging purposes
Main scripts: ddpm.py, humans_dataset_ldm.py, main.py, affordances.yaml, 3_30_iterate_captions.py


inference
inpaint_hic_sd.py ( launch with inference.sh)
	instructions: select MODEL_PATH, DATA_PATH, INF_MODE, CFG_SCALE
	


plotting
3_23_plot_joint_losses.py (compare two Kulal run loss curves same plot)
3_21_plot_losses.py (plot loss from TestTube output for a single given result run)

key runs


output_837436.txt - augmentations, full
output_838212.txt- no augmentations, full

modules
affordance-insertion/ldm/models/diffusion/ddpm.py
	has details on exact processes of ddpm in repo

affordance-insertion/main.py
	mostly loads configs and instantiates things from configs/ starts straining

main_slurm_launch.sh or train.sh
	launch training on slurm cluster

data generation

	preprocessing data into Kulal pipeline format
	Kulal Data Preprocessing


	downloading NTU dataset
/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/data/NTU_RGBD/unzip_downloads.py
		unzip NTURGB/ similar
	hallucinating-scenes/data/NTU_RGBD/download.sh
follow instructions elsewhere to obtain cookies (daily note), then should last the whole day and can download sequential zips to slurm
	download_bash_slurm_wrapper.sh
		wraps the above script in environment to persist in slurm
	
	caption generation

run_caption_gen_long_4_1.sh
	launch script for below
3_30_iterate_captions_v2.py
	generated:
		kulal_kinetics_composites_val.json
		kulal_kinetics_composites_train.json
	sends samples to
		kulal_kinetics_composites_train

hallucinating-scenes/data/example_captions_for_gpt/example_captions.json
same directory contains captions example images too





extended modules

/coc/flash3/nwarner30/image_editing/affordance-insertion/ldm/data/data_hic/humans_dataset_ldm.py
	contains dataset construction and augmentation operations used as well as dataloader.getitem() function

/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py
	look at .advance() criteria
	contains criteria for an epoch level update

/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py
	look at .advance() criteria
	should contain criteria for a single batch update
	

/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/loops/fit_loop.py


/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py
	contains main parent scripts for pytorch lightning level trainer

/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/loops/base.py
	contains loop scripts, unsure how differs from other loop py files but maybe more generic?

/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/loops/optimization/optimizer_loop.py
	manages accumulated gradient handling for optimization criteria for backwards pass
	something about tied hooks at bottom

/srv/essa-lab/flash3/nwarner30/miniconda3/envs/affordance_v2/lib/python3.8/site-packages/pytorch_lightning/core/optimizer.py
	contains the specific optimizer init configs
	however, actual optimizer initialization happens within main.py and the ddpm.py script

affordance-insertion/ldm/modules/encoders/modules.py
	manages the different encoders that we use, and modifications to retrieve overall embedding versus last hidden state.

why wont ipdb stop in subscript?

modules to compare to

hallucinating-scenes/scripts/2_18_training_captions_fid_etc.py
	standard SOTA latest script with eval features from our previous months of work.


misc


