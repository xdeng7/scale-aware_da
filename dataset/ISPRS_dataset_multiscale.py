import os
import os.path as osp
import random
import numpy as np
import torch
from skimage import io

# This code is strongly borrowed from 


# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

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

class ISPRS_dataset_multi(torch.utils.data.Dataset):
    def __init__(self,scale, ids, DATA_FOLDER1, LABEL_FOLDER1,DATA_FOLDER2, LABEL_FOLDER2,cache=False,WINDOW_SIZE=(256,256),augmentation=True):
        super(ISPRS_dataset_multi, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.window_size= WINDOW_SIZE
        
        # List of files
        
        self.data_files1 = [DATA_FOLDER1.format(id) for id in ids]
        self.label_files1 = [LABEL_FOLDER1.format(id) for id in ids]

        self.data_files2 = [DATA_FOLDER2.format(id) for id in ids]
        self.label_files2 = [LABEL_FOLDER2.format(id) for id in ids]


        # Sanity check : raise an error if some files do not exist
        for f in self.data_files1 + self.label_files1:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_1 = {}
        self.label_cache_1 = {}
        self.data_cache_2={}
        self.label_cache_2={}
        self.scale=scale

            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 60000*5
    
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
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files1) - 1)

        # print('random_idx:{}'.format(random_idx))
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_1.keys():
            data1 = self.data_cache_1[random_idx]
            data2 = self.data_cache_2[random_idx]
        else:
            # Data is normalized in [0, 1]
            data1 = 1/255 * np.asarray(io.imread(self.data_files1[random_idx]).transpose((2,0,1)), dtype='float32')
            data2 = 1/255 * np.asarray(io.imread(self.data_files2[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_1[random_idx] = data1
                self.data_cache_2[random_idx] = data2
            
        if random_idx in self.label_cache_1.keys():
            label1 = self.label_cache_1[random_idx]
            label2 = self.label_cache_2[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label1 = np.asarray(convert_from_color(io.imread(self.label_files1[random_idx])), dtype='int64')
            label2 = np.asarray(convert_from_color(io.imread(self.label_files2[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_1[random_idx] = label1
                self.label_cache_2[random_idx] = label2

        # Get a random patch
        scale=self.scale
        W, H = data2.shape[-2:]
        x1, x2, y1, y2 = get_random_pos(data1, self.window_size)

        # import pdb
        # pdb.set_trace()

        x0=int((x1/scale+x2/scale)/2)
        y0=int((y1/scale+y2/scale)/2)
        h=int(self.window_size[0]/2)
        w=int(self.window_size[1]/2)
        x1_, x2_, y1_, y2_ = int(x0-w), int(x0+w), int(y0-h), int(y0+h)

        while x1_<0 or x1_+self.window_size[0]>=W or y1_<0 or y1_+self.window_size[1]>=H:
            x1, x2, y1, y2 = get_random_pos(data1, self.window_size)

            x0=int((x1/scale+x2/scale)/2)
            y0=int((y1/scale+y2/scale)/2)
            x1_, x2_, y1_, y2_ = int(x0-w), int(x0+w), int(y0-h), int(y0+h)


        data_p1 = data1[:, x1:x2,y1:y2]
        label_p1 = label1[x1:x2,y1:y2]

        data_p2 = data2[:, x1_:x2_,y1_:y2_]
        label_p2 = label2[x1_:x2_,y1_:y2_]
        
        # Data augmentation
        # data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        # return (torch.from_numpy(data_p),
        #         torch.from_numpy(label_p))

        # print('%d %d %d %d'%(x1_,x2_,y1_,y2_))


        return (data_p2,label_p2, data_p1,  label_p1 )


