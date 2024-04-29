import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# User-defined labels for the configurations
config_1_label = 'Default Pose Embedding'  # Replace with the actual label name
config_2_label = 'Default + Spatial Heatmap'  # Replace with the actual label name

FIG_NAME = '4_26_comparison.png'

# Paths to the CSV files

csv_file_path_config_1 = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-04-21T22-51-01_affordance/testtube/version_0/metrics.csv'  # Replace with actual path
csv_file_path_config_2 = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-04-23T13-04-38_affordance/testtube/version_0/metrics.csv'  # Replace with actual path


# Reading the CSV data directly from the files
data_config_1 = pd.read_csv(csv_file_path_config_1)
data_config_2 = pd.read_csv(csv_file_path_config_2)

# Use forward fill to propagate the last valid values forward
data_config_1.ffill(inplace=True)
data_config_2.ffill(inplace=True)

EMA = False

# Isolate the columns of interest and rename for clarity
if EMA:
    data_config_1 = data_config_1[['epoch', 'val/loss_ema', 'train/loss_epoch']].rename(columns={'val/loss_ema': f'val/loss_{config_1_label.lower()}_ema', 'train/loss_epoch': f'train/loss_{config_1_label.lower()}_epoch'})
    data_config_2 = data_config_2[['epoch', 'val/loss_ema', 'train/loss_epoch']].rename(columns={'val/loss_ema': f'val/loss_{config_2_label.lower()}_ema', 'train/loss_epoch': f'train/loss_{config_2_label.lower()}_epoch'})
else:
    data_config_1 = data_config_1[['epoch', 'val/loss', 'train/loss_epoch']].rename(columns={'val/loss': f'val/loss_{config_1_label.lower()}', 'train/loss_epoch': f'train/loss_{config_1_label.lower()}_epoch'})
    data_config_2 = data_config_2[['epoch', 'val/loss', 'train/loss_epoch']].rename(columns={'val/loss': f'val/loss_{config_2_label.lower()}', 'train/loss_epoch': f'train/loss_{config_2_label.lower()}_epoch'})

# Merging dataframes on epoch
merged_data = pd.merge(data_config_1, data_config_2, on='epoch')

# Plotting the training and validation loss for both configurations
plt.figure(figsize=(14, 8))

# Apply a rolling mean with a window size, for example 10.
window_size = 20 # window size 1 = no smoothing
merged_data[f'train/loss_{config_1_label.lower()}_epoch_smoothed'] = merged_data[f'train/loss_{config_1_label.lower()}_epoch'].rolling(window=window_size).mean()
merged_data[f'train/loss_{config_2_label.lower()}_epoch_smoothed'] = merged_data[f'train/loss_{config_2_label.lower()}_epoch'].rolling(window=window_size).mean()
merged_data[f'val/loss_{config_1_label.lower()}_smoothed'] = merged_data[f'val/loss_{config_1_label.lower()}'].rolling(window=window_size).mean()
merged_data[f'val/loss_{config_2_label.lower()}_smoothed'] = merged_data[f'val/loss_{config_2_label.lower()}'].rolling(window=window_size).mean()

# Now replace the original plot lines with the smoothed data
plt.plot(merged_data['epoch'], merged_data[f'train/loss_{config_1_label.lower()}_epoch_smoothed'], label=f'Training Loss {config_1_label}', linestyle='-')
plt.plot(merged_data['epoch'], merged_data[f'train/loss_{config_2_label.lower()}_epoch_smoothed'], label=f'Training Loss {config_2_label}', linestyle='-')
plt.plot(merged_data['epoch'], merged_data[f'val/loss_{config_1_label.lower()}_smoothed'], label=f'Validation Loss {config_1_label}', linestyle='-')
plt.plot(merged_data['epoch'], merged_data[f'val/loss_{config_2_label.lower()}_smoothed'], label=f'Validation Loss {config_2_label}', linestyle='-')

plt.title(f'Comparison of Training and Validation Loss for {config_1_label} vs {config_2_label}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 0.1)  # Set the limit of the y-axis to max 0.1
#plt.ylim(0, 1.0)  # Set the limit of the y-axis to max 0.1
#plt.xlim(0, 20.0)  # Set the limit of the y-axis to max 0.1
plt.savefig("%s" % FIG_NAME)
plt.show()
