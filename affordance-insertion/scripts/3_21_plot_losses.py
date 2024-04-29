import pandas as pd
import matplotlib.pyplot as plt
import ipdb

# Specify the path to your CSV file
# Base run with augmentations enabled
csv_file_path = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-03-20T10-51-10_affordance/testtube/version_0/metrics.csv'

# Run with augmentations disabled (starting criteria similar for some reason)
csv_file_path = '/srv/essa-lab/flash3/nwarner30/image_editing/affordance-insertion/logs/2024-03-21T17-51-52_affordance/testtube/version_0/metrics.csv'

# Reading the CSV data directly from the file
data = pd.read_csv(csv_file_path)

# Use forward fill to propagate the last valid values forward
data.ffill(inplace=True)

EMA = False

# Isolate the columns of interest
if EMA:
    data = data[['epoch', 'val/loss_ema', 'train/loss_epoch']]
else:
    data = data[['epoch', 'val/loss', 'train/loss_epoch']]

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))

plt.plot(data['epoch'], data['train/loss_epoch'], label='Training Loss (Epoch)', linestyle='-')

if EMA:
    plt.plot(data['epoch'], data['val/loss_ema'], label='Validation Loss EMA', linestyle='-')
else:
    plt.plot(data['epoch'], data['val/loss'], label='Validation Loss', linestyle='-')


plt.title('Training vs Validation Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
#plt.ylim(0, 0.1)  # Set the limit of the y-axis to max 0.1
plt.savefig("3_23_test_kulal_our_filtered_kinetics_no_EMA.png")
plt.show()
