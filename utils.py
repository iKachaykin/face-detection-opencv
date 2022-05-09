import os
import glob

import numpy as np


def reshape_flattened_output(
        flattened_output, frames_indices, batch_size, output_shape=-1
):
    reshaped_detections = []
    zeros_shape = __get_zeros_shape(output_shape)
    for i in np.arange(batch_size):
        mask = np.array(frames_indices == i)
        if len(mask) > 0:
            reshaped_detections.append(flattened_output[mask])
        else:
            reshaped_detections.append(np.zeros(zeros_shape))
    return reshaped_detections


def __get_zeros_shape(output_shape):
    if isinstance(output_shape, tuple):
        zeros_shape = (0,) + output_shape
    elif output_shape > 0:
        zeros_shape = (0, output_shape)
    else:
        zeros_shape = (0,)
    return zeros_shape


def extract_paths_by_extensions(directory, extensions):
    extracted_paths = []
    for single_ext in extensions:
        single_glob_arg = os.path.join(directory, '**', f'*.{single_ext}')
        single_extracted_paths = glob.glob(single_glob_arg, recursive=True)
        extracted_paths.extend(single_extracted_paths)
    return extracted_paths
