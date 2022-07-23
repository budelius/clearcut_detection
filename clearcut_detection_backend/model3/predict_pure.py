import os
import re
import logging
import rasterio
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from rasterio.windows import Window
import cv2
from skimage.exposure import match_histograms
import segmentation_models_pytorch as smp
from torch import nn
import matplotlib.pyplot as plt

img_current = "/Users/pseekoo/Documents/data/sentinel/32TPT/2022-04-21/32TPT_2022-04-21_output.tif"
img_previous = "/Users/pseekoo/Documents/data/sentinel/32TPT/2022-05-11/32TPT_2022-05-11_output.tif"

filename = re.split(r'[./]', img_current)[-2] + ".tif"

save_path = "~/Documents/data/out"

input_size = 56
neighbours = 3
network = "unet_ch"
model_weights_path = "/Volumes/GoogleDrive/My Drive/data/models/unet_diff.pth"
channels = ['rgb', 'b8', 'b8a', 'b11', 'b12', 'ndvi', 'ndmi']
#channels = ['rgb', 'ndvi', 'ndmi']

def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch in ['ndvi', 'ndmi', 'b8', 'b8a', 'b11', 'b12']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count

def mask_postprocess(mask):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    return closing

def reshape_as_image(arr):
    """Returns the source array reshaped into the order
    expected by image processing and visualization software
    (matplotlib, scikit-image, etc)
    by swapping the axes order from (bands, rows, columns)
    to (rows, columns, bands)

    Parameters
    ----------
    arr : array-like of shape (bands, rows, columns)
        image to reshape
    """
    # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
    im = np.ma.transpose(arr, [1, 2, 0])
    return im

def get_model(name, classification_head=True, model_weights_path=None):
    if name == 'unet_ch':
        aux_params = dict(
            pooling='max',             # one of 'avg', 'max'
            dropout=0.1,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        return smp.Unet('resnet18', aux_params=aux_params,
                        encoder_weights=None, encoder_depth=2,
                        decoder_channels=(256, 128))
    else:
        raise ValueError("Unknown network")

logging.info(f'network:{network}')
logging.info(f'model_weights_path: {model_weights_path}')
logging.info(f'channels: {channels}')
logging.info(f'neighbours: {neighbours}')

model = get_model(network)

device = torch.device("cpu")
map_location = device
if torch.cuda.is_available():
    device = torch.device("cuda")
    map_location = torch.device(device)
if hasattr(torch, 'has_mps'):
    if torch.has_mps:
        device = torch.device("mps")
        map_location = torch.device(torch.device("cpu"))

model.encoder.conv1 = nn.Conv2d(
    count_channels(channels) * neighbours, 64, kernel_size=(7, 7),
    stride=(2, 2), padding=(3, 3), bias=False
)

model_weights = torch.load(model_weights_path, map_location=map_location)
model.load_state_dict(model_weights)

with rasterio.open(img_current) as source_current, \
        rasterio.open(img_previous) as source_previous:
    meta = source_current.meta
    meta['count'] = 1
    clearcut_mask = np.zeros((source_current.height, source_current.width))
    for i in tqdm(range(source_current.width // input_size)):
        for j in range(source_current.height // input_size):
            bottom_row = j * input_size
            upper_row = (j + 1) * input_size
            left_column = i * input_size
            right_column = (i + 1) * input_size

            corners = [
                source_current.xy(bottom_row, left_column),
                source_current.xy(bottom_row, right_column),
                source_current.xy(upper_row, right_column),
                source_current.xy(upper_row, left_column),
                source_current.xy(bottom_row, left_column)
            ]

            window = Window(bottom_row, left_column, input_size, input_size)
            image_current = reshape_as_image(source_current.read(window=window))
            image_previous = reshape_as_image(source_previous.read(window=window))

            image_diff = match_histograms(image_current, image_previous, multichannel=True)

            cv2.imwrite(f'{save_path}.png', image_diff)

            # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(56, 56),
            #                                     sharex=True, sharey=True)
            # for aa in (ax1, ax2, ax3):
            #     aa.set_axis_off()

            # ax1.imshow(image_current)
            # ax1.set_title('Current')
            # ax2.imshow(image_previous)
            # ax2.set_title('Previous')
            # ax3.imshow(image_diff)
            # ax3.set_title('Matched')

            difference = ((image_current - image_diff) / (image_current + image_diff))
            difference = (difference + 1) * 127
            difference_image = np.concatenate((difference.astype(np.uint8), image_current.astype(np.uint8), image_diff.astype(np.uint8)), axis=-1)

            image_tensor = transforms.ToTensor()(difference_image.astype(np.uint8)).to(device, dtype=torch.float)

            image_shape = 1, count_channels(channels) * neighbours, input_size, input_size
            prediction, _ = model.to(device).predict(image_tensor.view(image_shape).to(device, dtype=torch.float))
            predicted = prediction.view(input_size, input_size).detach().cpu().numpy()

            predicted = mask_postprocess(predicted)
            clearcut_mask[left_column:right_column, bottom_row:upper_row] += predicted

meta['dtype'] = 'float32'
raster_array = clearcut_mask.astype(np.float32)
# raster_array, meta = predict_raster(
#             image_path_current,
#             image_path_previous,
#             channels, network, model_weights_path
#         )

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
    logging.info("Data directory created.")

save_path = os.path.join(save_path, f'predicted_{filename}')

cv2.imwrite(f'{save_path}.png', raster_array)

with rasterio.open(f'{save_path}.tif', 'w', **meta) as dst:
    for i in range(1, meta['count'] + 1):
        dst.write(raster_array, i)