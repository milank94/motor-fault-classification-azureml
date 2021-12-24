import argparse
import glob
import os

import joblib
import numpy as np
from azureml.core import Run
from utils.acquisition_functions import read_data
from utils.processing_functions import (compute_fft, down_sample_data,
                                        scale_data)

# Get the experiment run context
run = Run.get_context()

# Set parameters
parser = argparse.ArgumentParser()
parser.add_argument('--start-index', type=int, dest='start_index', default=0)
parser.add_argument('--resample-rate', type=int, dest='resample_rate', default=100)
parser.add_argument('--out-folder', type=str, dest='folder')
parser.add_argument('--ds-normal', type=str, dest='ds_ref_normal')
# parser.add_argument('--ds-vertical', type=str, dest='ds_ref_vertical')
# parser.add_argument('--ds-horizontal', type=str, dest='ds_ref_horizontal')
# parser.add_argument('--ds-imbalance', type=str, dest='ds_ref_imbalance')
# parser.add_argument('--ds-overhang', type=str, dest='ds_ref_overhang')
# parser.add_argument('--ds-underhang', type=str, dest='ds_ref_underhang')
args = parser.parse_args()
start_index = args.start_index
resample_rate = args.resample_rate
output_folder = args.folder

# Prepare the dataset
data_normal = np.stack(read_data(glob.glob(args.ds_ref_normal + '/*.csv')))
# data_horizontal = np.stack(read_data(glob.glob(args.ds_ref_horizontal + '/*.csv')))
# data_vertical = np.stack(read_data(glob.glob(args.ds_ref_vertical + '/*.csv')))
# data_imbalance = np.stack(read_data(glob.glob(args.ds_ref_imbalance + '/*.csv')))
# data_overhang = np.stack(read_data(glob.glob(args.ds_ref_overhang + '/*.csv')))
# data_underhang = np.stack(read_data(glob.glob(args.ds_ref_underhang + '/*.csv')))

y_1 = np.zeros(int(len(data_normal)), dtype=int)
# y_2 = np.full(int(len(data_horizontal)), 1)
# y_3 = np.full(int(len(data_imbalance)), 2)
# y_4 = np.full(int(len(data_vertical)), 3)
# y_5 = np.full(int(len(data_overhang)), 4)
# y_6 = np.full(int(len(data_underhang)), 5)
# y = np.concatenate((y_1, y_2, y_3, y_4, y_5, y_6))

# X = np.concatenate((data_normal, data_horizontal, data_imbalance, data_vertical, data_overhang, data_underhang))

# Temp
y = np.concatenate((y_1))
X = np.concatenate((data_normal))

resampled_data = down_sample_data(X, start_index, resample_rate)
scaled_data, norm = scale_data(resampled_data)
prepped_X = compute_fft(scale_data)

# Save the fitted scaler
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=norm, filename='outputs/scaler.pkl')

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
joblib.dump(value=(prepped_X, y), filename=f'{output_folder}/prepped_data.pkl')
