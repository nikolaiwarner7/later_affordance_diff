import numpy as np
import ipdb

class DatasetSplitChecker:
    def __init__(self, data_split):
        self.data_split = data_split

    def check_overlap(self):
        # Load train split video names
        train_video_names = set(np.load('/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/2_23_kinetics_train_training_videos.npy', allow_pickle=True).tolist())
        # Load validation split video names
        val_video_names = set(np.load('/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/2_23_kinetics_train_validation_all_captioned_videos.npy', allow_pickle=True).tolist())

        ipdb.set_trace()
        # Check and print any overlap
        if self.data_split == 'train':
            overlap = train_video_names.intersection(val_video_names)
        elif self.data_split == 'val':
            overlap = val_video_names.intersection(train_video_names)
        
        if overlap:
            print("Overlap detected:", overlap)
        else:
            print("No overlap detected between the splits.")

# Example usage
checker = DatasetSplitChecker(data_split='train')
checker.check_overlap()
