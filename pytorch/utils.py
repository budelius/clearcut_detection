import os
import re

import imageio
import numpy as np


def join_paths(*paths):
    return os.path.join(*paths)


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_paths(*path_parts), file_type)


def read_tensor(filepath):
    return imageio.imread(filepath)


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch in ['ndvi', 'b8']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def filter_by_channels(image_tensor, channels):
    result = []
    for ch in channels:
        if ch == 'rgb':
            result.append(image_tensor[:, :, :3])
        elif ch == 'ndvi':
            result.append(image_tensor[:, :, 3:4])
        elif ch == 'b8':
            result.append(image_tensor[:, :, 4:5])
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return np.concatenate(result, axis=2)


def get_image_info(instance):
    name_parts = re.split(r'[_.]', instance)
    return '_'.join(name_parts[:2]), '_'.join(name_parts[-3:-1])
