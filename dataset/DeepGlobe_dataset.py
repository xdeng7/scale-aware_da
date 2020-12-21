import os
import os.path as osp
import random
import numpy as np
import torch
from skimage import io

# This code is strongly borrowed from 

# Label
# Each satellite image is paired with a mask image for land cover annotation. The mask is a RGB image with 7 classes of labels, using color-coding (R, G, B) as follows.
# Urban land: 0,255,255 - Man-made, built up areas with human artifacts (can ignore roads for now which is hard to label)
# Agriculture land: 255,255,0 - Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations.
# Rangeland: 255,0,255 - Any non-forest, non-farm, green land, grass
# Forest land: 0,255,0 - Any land with x% tree crown density plus clearcuts.
# Water: 0,0,255 - Rivers, oceans, lakes, wetland, ponds.
# Barren land: 255,255,255 - Mountain, land, rock, dessert, beach, no vegetation
# Unknown: 0,0,0 - Clouds and others

# DeepGlobe color palette
# Let's define the standard DeepGlobe color palette
palette = {0 : (0,255,255), # Urban land (cyan)
           1 : (255,255,0),     # Agriculture land (yellow)
           2 : (255,0,255),   # Rangeland (purple)
           3 : (0, 255, 0),     # Forest land (green)
           4 : (0,0,255),   # Water (blue)
           5 : (255,255,255),     # Barren land (white)
           6 : (0,0,0)}       # Unknown (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

class DeepGlobe_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, DATA_FOLDER, LABEL_FOLDER, cache=False,WINDOW_SIZE=(256,256),augmentation=True):
        super(DeepGlobe_dataset, self).__init__()
        
        self.augmentation = augmentation

        
        # List of files
        
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts

        self.cache=cache
        self.window_size=WINDOW_SIZE


        self.data_cache_ = {}
        self.label_cache_ = {}

            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 250000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, index):
        # Pick a random image

        random_idx = random.randint(0, len(self.data_files) - 1)

        # print('random_idx:{}'.format(random_idx))
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)


        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


