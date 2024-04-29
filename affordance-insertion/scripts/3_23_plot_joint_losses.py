import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the CSV files
#csv_file_path_aug = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-03-20T10-51-10_affordance/testtube/version_0/metrics.csv'
#csv_file_path_no_aug = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-03-21T17-51-52_affordance/testtube/version_0/metrics.csv'

# Reading the CSV data directly from the files
data_aug = pd.read_csv(csv_file_path_aug)
data_no_aug = pd.read_csv(csv_file_path_no_aug)

# Use forward fill to propagate the last valid values forward
data_aug.ffill(inplace=True)
data_no_aug.ffill(inplace=True)

EMA = False

# Isolate the columns of interest and rename for clarity
if EMA:
    data_aug = data_aug[['epoch', 'val/loss_ema', 'train/loss_epoch']].rename(columns={'val/loss_ema': 'val/loss_aug_ema', 'train/loss_epoch': 'train/loss_aug_epoch'})
    data_no_aug = data_no_aug[['epoch', 'val/loss_ema', 'train/loss_epoch']].rename(columns={'val/loss_ema': 'val/loss_no_aug_ema', 'train/loss_epoch': 'train/loss_no_aug_epoch'})
else:
    data_aug = data_aug[['epoch', 'val/loss', 'train/loss_epoch']].rename(columns={'val/loss': 'val/loss_aug', 'train/loss_epoch': 'train/loss_aug_epoch'})
    data_no_aug = data_no_aug[['epoch', 'val/loss', 'train/loss_epoch']].rename(columns={'val/loss': 'val/loss_no_aug', 'train/loss_epoch': 'train/loss_no_aug_epoch'})

# Merging dataframes on epoch
merged_data = pd.merge(data_aug, data_no_aug, on='epoch')


# Plotting the training and validation loss for both runs
plt.figure(figsize=(14, 8))

# Apply a rolling mean with a window size, for example 10.
window_size = 30
merged_data['train/loss_aug_epoch_smoothed'] = merged_data['train/loss_aug_epoch'].rolling(window=window_size).mean()
merged_data['train/loss_no_aug_epoch_smoothed'] = merged_data['train/loss_no_aug_epoch'].rolling(window=window_size).mean()
merged_data['val/loss_aug_smoothed'] = merged_data['val/loss_aug'].rolling(window=window_size).mean()
merged_data['val/loss_no_aug_smoothed'] = merged_data['val/loss_no_aug'].rolling(window=window_size).mean()

# Now replace the original plot lines with the smoothed data
plt.plot(merged_data['epoch'], merged_data['train/loss_aug_epoch_smoothed'], label='Training Loss Augmented', linestyle='-')
plt.plot(merged_data['epoch'], merged_data['train/loss_no_aug_epoch_smoothed'], label='Training Loss No Aug', linestyle='-')
plt.plot(merged_data['epoch'], merged_data['val/loss_aug_smoothed'], label='Validation Loss Augmented', linestyle='-')
plt.plot(merged_data['epoch'], merged_data['val/loss_no_aug_smoothed'], label='Validation Loss No Aug', linestyle='-')




"""
# Training loss
plt.plot(merged_data['epoch'], merged_data['train/loss_aug_epoch'], label='Training Loss Augmented', linestyle='-')
plt.plot(merged_data['epoch'], merged_data['train/loss_no_aug_epoch'], label='Training Loss No Aug', linestyle='-')

# Validation loss
if EMA:
    plt.plot(merged_data['epoch'], merged_data['val/loss_aug_ema'], label='Validation Loss Augmented EMA', linestyle='--')
    plt.plot(merged_data['epoch'], merged_data['val/loss_no_aug_ema'], label='Validation Loss No Aug EMA', linestyle='--')
else:
    plt.plot(merged_data['epoch'], merged_data['val/loss_aug'], label='Validation Loss Augmented', linestyle='--')
    plt.plot(merged_data['epoch'], merged_data['val/loss_no_aug'], label='Validation Loss No Aug', linestyle='--')
"""
plt.title('Comparison of Training and Validation Loss for Augmented vs Non-Augmented Runs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 0.1)  # Set the limit of the y-axis to max 0.1
plt.savefig("3_23_joint_plot_train_val_loss_comparison.png")
plt.show()
