import os
import cv2
import csv
import glob
import json
import time
import random
import shutil
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from tqdm import tqdm

import openslide
import staintools
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Colormap
from skimage import morphology
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Resizing
from tensorflow.keras.utils import Sequence


from concurrent.futures import ThreadPoolExecutor, as_completed
import functools



def get_TC(weights, image_size=(512, 512), num_classes=3, output_features=False):
    initializer = tf.keras.initializers.GlorotNormal()
    net = MobileNet(include_top=False, input_shape=(image_size[0], image_size[1], 3))
    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='Dense_1', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='Dense_2', kernel_initializer=initializer)(x)
    feature_layer = x
    output_layer = Dense(num_classes, activation='softmax', name='Predictions')(x)

    model = Model(inputs=net.input, outputs=feature_layer if output_features else output_layer)
    model.load_weights(weights)
    return model



def get_wsiname(file):
    if ".mrxs" in file:
        return os.path.basename(file).split(".mrxs")[0]
    elif ".ndpi" in file:
        return os.path.basename(file).split(".ndpi")[0]
    elif ".svs" in file:
        return os.path.basename(file).split(".svs")[0]
    return os.path.basename(file)



def parse_patch_size(wsi, patch_size_microns=128):
    mpp_x, mpp_y = float(wsi.properties['openslide.mpp-x']), float(wsi.properties['openslide.mpp-y'])
    patch_size_x, patch_size_y = int(patch_size_microns // mpp_x), int(patch_size_microns // mpp_y)
    return patch_size_x, patch_size_y




def Reinhard(img_arr, standard_img="./data/he.jpg"):
    target = staintools.read_image(standard_img)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target)
    img_to_transform = staintools.LuminosityStandardizer.standardize(img_arr)
    return normalizer.transform(img_to_transform)




def process_tile_static(wsi, image_level, x, y, patch_size, stride, level_multiplier, downsample):
    try:
        image_tile = wsi.read_region((int(x * level_multiplier), int(y * level_multiplier)), image_level, (patch_size, patch_size)).convert("RGB")
        image_tile = np.array(image_tile).astype('uint8')
        image_tile = Reinhard(image_tile)
        
        if downsample != 1:
            image_tile = image_tile[::downsample, ::downsample, :]
            
        return image_tile, x // stride, y // stride
        
    except Exception as e:
        print(f"Error processing tile at ({x}, {y}) in {wsi_path}: {e}")
        return None




class SlideIterator:
    def __init__(self, wsi, image_level=0, mask_path=None, threshold_mask=0.8):
        self.image = wsi  
        self.image_level = image_level
        self.image_shape = self.image.level_dimensions[image_level]
        self.image_level_multiplier = self.image.level_dimensions[0][0] // self.image.level_dimensions[1][0]

        if mask_path:
            try:
                self.mask = np.transpose(np.load(mask_path), axes=(1, 0))
                self.mask_shape = self.mask.shape
            except Exception as e:
                raise ValueError(f"Error loading mask from {mask_path}: {e}")
        else:
            self.mask = None
            self.mask_shape = None
        
        self.threshold_mask = threshold_mask
        self.image_mask_ratio = 1 if self.mask is None else int(self.image_shape[0] / self.mask_shape[0])

    
    def get_patch_coords(self, patch_size, stride):
        coords = []
        self.coords_shape = self.image_shape[0] // stride, self.image_shape[1] // stride
        patch_ratio = patch_size // self.image_mask_ratio

        for index_y in range(0, self.image_shape[1], stride):
            for index_x in range(0, self.image_shape[0], stride):
                if (index_x // stride >= self.coords_shape[0]) or (index_y // stride >= self.coords_shape[1]):
                    continue

                if self.mask is not None:
                    mask_x, mask_y = int(index_x / self.image_mask_ratio), int(index_y / self.image_mask_ratio)
                    mask_tile = self.mask[mask_x: mask_x + patch_ratio, mask_y: mask_y + patch_ratio].flatten()
                else:
                    mask_tile = None

                if mask_tile is None or mask_tile.mean() > self.threshold_mask:
                    coords.append((index_x, index_y))

        return coords, self.coords_shape

    
    def save_array(self, patch_size, stride, output_pattern, downsample=1, use_multithreading=True, max_workers=None):
        filename = os.path.splitext(os.path.basename(output_pattern))[0]
        coords, coords_shape = self.get_patch_coords(patch_size, stride)
        level_multiplier = self.image_level_multiplier ** self.image_level

        process_fn = functools.partial(
            process_tile_static,
            self.image,
            self.image_level,
            patch_size=patch_size,
            stride=stride, 
            level_multiplier=level_multiplier,
            downsample=downsample
        )

        results = []

        if use_multithreading:
            if max_workers is None:
                max_workers = os.cpu_count()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                with tqdm(total=len(coords), desc=f"Extracting patches from {filename}") as pbar:
                    futures = {executor.submit(process_fn, x, y): (x, y) for x, y in coords}

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error processing patch at {futures[future]}: {e}")
        else:
            with tqdm(total=len(coords), desc=f"Extracting patches from {filename}") as pbar:
                for x, y in coords:
                    try:
                        result = process_fn(x, y)
                        if result is not None:
                            results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing patch at ({x}, {y}): {e}")

        if results:
            image_tiles, xs, ys = zip(*results)
            image_tiles = np.stack(image_tiles, axis=0).astype("uint8")
            xs, ys = np.array(xs), np.array(ys)

            np.save(f"{output_pattern}_patches.npy", image_tiles)
            np.save(f"{output_pattern}_x_idx.npy", xs)
            np.save(f"{output_pattern}_y_idx.npy", ys)
            np.save(f"{output_pattern}_im_shape.npy", coords_shape)

            print(f"Tiles extracted for {filename}: {image_tiles.shape[0]} patches saved.")
        else:
            print(f"No valid patches found for {filename}.")




def vectorise_wsi(wsi, mask_path, patch_size, foreground_thes, output_pattern, use_multithreading, max_workers):
    patches_file = output_pattern + '_patches.npy'
    if os.path.exists(patches_file):
        print(f"Pattern files for {output_pattern.split('_pattern')[0]} already exist. Skipping preprocessing.")
        return None  # Return None or the existing SlideIterator if needed
    
    print(f"Preprocessing {output_pattern.split('_pattern')[0]}...")
    si = SlideIterator(wsi=wsi, image_level=0, mask_path=mask_path, threshold_mask=foreground_thes)
    si.save_array(patch_size=patch_size, stride=patch_size, output_pattern=output_pattern, downsample=1, use_multithreading=use_multithreading, max_workers=max_workers)
    
    print(f"Vectorization completed for {output_pattern.split('_pattern')[0]}.")
    return si




class WsiNpySequence(keras.utils.Sequence):
    def __init__(self, wsi_pattern, batch_size, IMAGE_SIZE):
        self.batch_size = batch_size
        self.wsi_pattern = wsi_pattern
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_tiles = np.load(f'{wsi_pattern}_patches.npy')
        self.xs = np.load(f'{wsi_pattern}_x_idx.npy')
        self.ys = np.load(f'{wsi_pattern}_y_idx.npy')
        self.image_shape = np.load(f'{wsi_pattern}_im_shape.npy')

        self.n_samples = self.image_tiles.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        return batch

    def get_batch(self, idx):
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        image_tiles = self.image_tiles[idxs, ...]
        image_tiles = Resizing(self.IMAGE_SIZE[0], self.IMAGE_SIZE[1])(image_tiles)

        image_tiles = tf.cast(image_tiles, tf.float32)
        image_tiles = preprocess_input(image_tiles)

        return image_tiles, tf.ones((self.batch_size, 1), dtype=tf.float32) 



    
def predict_tc(model, wsi_sequence):
    print("Predicting tissue classes...")

    tc_predictions = model.predict(wsi_sequence, steps=len(wsi_sequence), verbose=1)  

    tissue_map = np.ones((wsi_sequence.image_shape[1], wsi_sequence.image_shape[0], tc_predictions.shape[1])) * np.nan
    
    for patch_feature, x, y in tqdm(zip(tc_predictions, wsi_sequence.xs, wsi_sequence.ys), total=len(wsi_sequence.xs), desc="Processing Patches"):
        tissue_map[y, x, :] = patch_feature
    
    tissue_map[np.isnan(tissue_map)] = 0
    return tissue_map




def clean_up_temp_files(TC_maskpt):
    output_dir = os.path.basename(TC_maskpt)
    for temp_file in glob.glob(f"{output_dir}/*pattern*"):
        os.remove(temp_file)
        print(f"{temp_file} removed!")


        

def run_TC_one_slide(wsi, mask_pt, save_pt, patch_size, foreground_thes=0.7, IMAGE_SIZE=None, model=None, free_space=False, use_multithreading=True, max_workers=8):
    if not os.path.exists(save_pt):
        mask_arr = np.array(Image.open(mask_pt))
        mask_path = mask_pt.replace("_mask_use.png", "_mask_use.npy")
        np.save(mask_path, (mask_arr / 255).astype("uint8"))
        print(f"Mask saved at {mask_path}.")
    
        output_pattern = save_pt.replace("_probmask.npy", "_pattern")
        vectorise_wsi(wsi, mask_path, patch_size, foreground_thes, output_pattern, use_multithreading, max_workers)

        wsi_sequence = WsiNpySequence(wsi_pattern=output_pattern, batch_size=8, IMAGE_SIZE=IMAGE_SIZE)
        
        tissue_map = predict_tc(model, wsi_sequence)
        np.save(save_pt, tissue_map)
        print(f"Tissue map saved at {save_pt}.")        

        if free_space:
            clean_up_temp_files(save_pt) 
            
    else:
        tissue_map = np.load(save_pt)
                  
    return tissue_map


    

def save_tc_map(tissue_map, wsi_path, save_pt):
    wsi = openslide.OpenSlide(wsi_path)
    if ".mrxs" in wsi_path:
        bounds_h = int(wsi.properties['openslide.bounds-height'])//512
        bounds_w = int(wsi.properties['openslide.bounds-width'])//512
        bounds_x = int(wsi.properties['openslide.bounds-x'])//512
        bounds_y = int(wsi.properties['openslide.bounds-y'])//512
        tissue_map = tissue_map[bounds_y:(bounds_y+bounds_h), bounds_x:(bounds_x+bounds_w),:]

    im = Image.fromarray((tissue_map * 255).astype("uint8"))
    im.save(save_pt)
    print(f"TC Map saved at {save_pt}.")

    
    

def save_Allpatch(tissue_map, patch_size, save_pt):
    patch_data = []
    for grid_x in range(tissue_map.shape[1]):
        for grid_y in range(tissue_map.shape[0]):
            TC_epi, TC_str, TC_adi = tissue_map[grid_y, grid_x, :]
            wsi_name = save_pt.split("/")[-2]
            patch_id = f"{wsi_name}_{grid_x}_{grid_y}_{patch_size}"
            orig_x, orig_y = int(grid_x * patch_size), int(grid_y * patch_size)
            cls = np.argmax([TC_epi, TC_str, TC_adi])
            cls = cls + 1
            if TC_epi==0:
                cls = 0
            patch_data.append([patch_id, grid_x, grid_y, orig_x, orig_y, TC_epi, TC_str, TC_adi, cls])
    
    patch_df = pd.DataFrame(patch_data, columns=["patch_id", "grid_x", "grid_y", "orig_x", "orig_y", "TC_epi", "TC_str", "TC_adi", "cls"])
    patch_df["cohort"] = save_pt.split("/")[-3]
    patch_df["wsi_id"] = save_pt.split("/")[-2]
    patch_df.to_csv(save_pt, index=False)
    print(f"{save_pt} saved!")
    return patch_df




def build_disrete_cmap(number=3):
    if number == 3:
        # 0:background; 1:epi; 2:stroma; 3:mixed 4: adi
        colors = np.array([
            [228, 26, 28],  # Red - epi
            [77, 167, 77],  # Green - stroma
            [128, 128, 128], # grey adipocytes
        ]) / 255

    else:
        colors = []
        cm = plt.get_cmap('gist_rainbow')
        for i in range(number):
            colors.append(cm(i//3*3.0/number))
        colors = np.array(colors)

    cmap = ListedColormap(colors, N=colors.shape[0])
    return cmap




def build_poly(tx: np.ndarray, ty: np.ndarray, bx: np.ndarray, by: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Counter clock-wise
    px = np.vstack((tx, bx, bx, tx)).T
    py = np.vstack((ty, ty, by, by)).T
    return px, py
    



def get_JSON(cls_df, json_pt, patch_size, require_bounds=False):
    tx = np.array(cls_df['orig_x']).astype("int")
    ty = np.array(cls_df['orig_y']).astype("int")
    bx = np.array([i+ patch_size for i in cls_df['orig_x']]).astype("int")
    by = np.array([i+ patch_size for i in cls_df['orig_y']]).astype("int")

    if require_bounds:
        bounds_x = int(wsi.properties['openslide.bounds-x'])
        bounds_y = int(wsi.properties['openslide.bounds-y'])
        tx = np.array(tx-bounds_x).astype("int")
        ty = np.array(ty-bounds_y).astype("int")
        bx = np.array(bx-bounds_x).astype("int")
        by = np.array(by-bounds_y).astype("int")

    names = list(cls_df['cls'])
    values = list(cls_df['cls'])

    # Build shape and simplify the shapes if True
    polys_x, polys_y = build_poly(tx=tx, ty=ty, bx=bx, by=by)
    cmap = build_disrete_cmap(number=4)
    
    coords = {}
    for i in range(len(polys_x)):

        color = 255 * np.array(cmap(int(values[i])))[:3]
        color = list(color) + [255]  
        coords[f'poly{i}'] = {
            "coords": np.vstack((polys_x[i], polys_y[i])).tolist(),
            "class": str(names[i]),
            "name": str(names[i]),
            "color": color  
        }

    if json_pt is not None:
        with open(json_pt, 'w') as outfile:
            json.dump(coords, outfile)
        print(f"{json_pt} saved!")


    
def process_TCmask(wsi, tissue_map, upsample, small_objects, roi_width):
    wsi_mask_ratio = wsi.level_dimensions[0][0] / tissue_map.shape[1] / upsample
    small_objects = small_objects / wsi_mask_ratio
    mpp = float(wsi.properties['openslide.mpp-x'])
    roi_width = int(roi_width / float(mpp) / wsi_mask_ratio) 
    
    epi_mask = tissue_map.copy()
    epi_mask = np.argmax(epi_mask, axis=-1)
    epi_mask[tissue_map[:,:,0]==0] = 99
    epi_mask[epi_mask != 0] = 99
    epi_mask[epi_mask == 0] = 1
    epi_mask[epi_mask == 99] = 0
    epi_mask = cv2.resize(np.uint8(epi_mask * 255), (epi_mask.shape[1]*upsample, epi_mask.shape[0]*upsample), interpolation=cv2.INTER_CUBIC) 
    epi_mask = np.array(epi_mask) > 0 
    
    # remove object area less than small_objects
    epi_mask = morphology.remove_small_objects(epi_mask.astype(bool), small_objects) 
    # fill small holes
    epi_mask = morphology.remove_small_holes(epi_mask.astype(bool), small_objects)
    epi_mask = epi_mask.astype("uint8")

    return epi_mask, roi_width, wsi_mask_ratio




def save_epi_mask(epi_mask, save_pt):
    epi_mask_im = Image.fromarray(epi_mask * 255)  # Convert to 255 (binary image)
    epi_mask_im.save(save_pt)
    print(f"{save_pt} saved!")



## save the JSON file
def get_roi_locs(mask, roi_width):
    roi = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    roi[:,:,0][mask] = np.random.randint(0,255)
    roi[:,:,1][mask] = np.random.randint(0,255)
    roi[:,:,2][mask] = np.random.randint(0,255)
    y_idx, x_idx, _ = np.nonzero(roi) 
    x1_inner, x2_inner = np.min(x_idx), np.max(x_idx)
    y1_inner, y2_inner = np.min(y_idx), np.max(y_idx)
    x1_outer = np.max((0, x1_inner-roi_width))
    y1_outer = np.max((0, y1_inner-roi_width))
    x2_outer = np.min((x2_inner+roi_width, roi.shape[1]))
    y2_outer = np.min((y2_inner+roi_width, roi.shape[0]))
    return x1_inner, y1_inner, x2_inner, y2_inner, x1_outer, y1_outer, x2_outer, y2_outer



def get_roi_ids(epi_mask, wsi_id, roi_width, wsi_mask_ratio, threshold=1):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(epi_mask, connectivity=8)
    roi_ids = []
    for i in range(1, num_labels):
        mask_i = labels == i
        _,_,_,_, x1_outer, y1_outer, x2_outer, y2_outer = get_roi_locs(mask_i, roi_width)
        # avoid getting too large ROIs
        # if (x2_outer-x1_outer <= mask_i.shape[1]//threshold) and (y2_outer-y1_outer <= mask_i.shape[0]//threshold):
        x1_outer, y1_outer, x2_outer, y2_outer =x1_outer*wsi_mask_ratio, y1_outer*wsi_mask_ratio, x2_outer*wsi_mask_ratio, y2_outer*wsi_mask_ratio
        roi_id = f"{wsi_id}_{int(x1_outer)}_{int(y1_outer)}_{int(x2_outer-x1_outer)}_{int(y2_outer-y1_outer)}"
        roi_ids.append(roi_id)
            
    print(f"There're {len(np.unique(roi_ids))} ROIs detected")
    return roi_ids



def roi_id2patch_id(roi_id, patch_size, tissue_map, patch_dict):
    roi_x, roi_y, roi_w, roi_h = roi_id.split("_")[-4:]
    roi_x, roi_y, roi_w, roi_h = int(roi_x), int(roi_y), int(roi_w), int(roi_h)
    grid_x1 = int(np.ceil(roi_x/patch_size))
    grid_y1 = int(np.ceil(roi_y/patch_size))
    grid_x2 = int(np.floor(roi_w/patch_size)) + grid_x1
    grid_y2 = int(np.floor(roi_h/patch_size)) + grid_y1
    roi_TC = tissue_map[grid_y1:grid_y2, grid_x1:grid_x2]
    # plt.imshow(roi_TC)
    roi_TC_cls = np.zeros((roi_TC.shape[0], roi_TC.shape[1])) 
    roi_TC_cls[roi_TC[:,:,0] > 0.5] = 1 # epi
    roi_TC_cls[roi_TC[:,:,1] > 0.5] = 2 # str
    roi_TC_cls[roi_TC[:,:,2] > 0.5] = 3 # adi
    # plt.imshow(roi_TC_cls)

    for cls in [1,2,3]:   
        ys, xs = np.nonzero(roi_TC_cls == cls) 
        for y, x in zip(ys, xs):
            # wsi grids
            grid_y, grid_x = y+grid_y1, x+grid_x1
            patch_id = f"{roi_id}_{grid_x}_{grid_y}_{patch_size}"
            patch_dict["roi_id"].append(roi_id)
            patch_dict["patch_id"].append(patch_id)
            patch_dict["cls"].append(cls)
            patch_dict["TC_epi"].append(tissue_map[grid_y, grid_x, 0])
            patch_dict["TC_str"].append(tissue_map[grid_y, grid_x, 1])
            patch_dict["TC_adi"].append(tissue_map[grid_y, grid_x, 2])



def getROIxy(patch_id):
    _,_,_,_,grid_x,grid_y, patch_size = patch_id.split("_")[-7:]
    grid_x, grid_y, patch_size = int(grid_x),int(grid_y),int(patch_size)
    x, y = grid_x * patch_size, grid_y * patch_size
    return x, y



def addWSIxy(df):
    x_orig = []
    y_orig = []
    for i in list(df["patch_id"]):
        x_orig_i, y_orig_i = getROIxy(i)
        x_orig.append(x_orig_i)
        y_orig.append(y_orig_i)
    df["orig_x"] = x_orig
    df["orig_y"] = y_orig
    return df



def save_ROIpatch(tissue_map, epi_mask, wsi_mask_ratio, roi_width, patch_size, save_pt):
    wsi_name = save_pt.split("/")[-2]
    roi_ids = get_roi_ids(epi_mask, wsi_name, roi_width, wsi_mask_ratio, threshold=1)
    patch_dict = {"roi_id": [], "patch_id": [], "cls": [], "TC_epi": [], "TC_str": [], "TC_adi": []}
    for roi_id in roi_ids:
        roi_id2patch_id(roi_id, patch_size, tissue_map, patch_dict)
    patch_df = pd.DataFrame.from_dict(patch_dict)
    patch_df["cohort"] = save_pt.split("/")[-3]
    patch_df["wsi_id"] = save_pt.split("/")[-2]
    patch_df = addWSIxy(patch_df)
    patch_df.to_csv(save_pt, index=False)
    print(f"Saved {save_pt}!")
    return patch_df




def bbx_overlay(epi_mask, overlay_path, roi_width, threshold=1):    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(epi_mask, connectivity=8)
    
    bbx_map = np.zeros((epi_mask.shape[0], epi_mask.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask_i = labels == i
        bbx_map[:,:,0][mask_i] = np.random.randint(0,255)
        bbx_map[:,:,1][mask_i] = np.random.randint(0,255)
        bbx_map[:,:,2][mask_i] = np.random.randint(0,255)
        
        x1_inner, y1_inner, x2_inner, y2_inner, x1_outer, y1_outer, x2_outer, y2_outer = get_roi_locs(mask_i, roi_width)
        # avoid getting too large ROIs
        if (x2_outer-x1_outer < mask_i.shape[1]//threshold) and (y2_outer-y1_outer < mask_i.shape[0]//threshold):
            bbx_map = cv2.rectangle(bbx_map, (x1_inner, y1_inner), (x2_inner,y2_inner), (255, 0, 0), 15)
            bbx_map = cv2.rectangle(bbx_map, (x1_outer, y1_outer), (x2_outer, y2_outer), (255, 255, 0), 15)

    plt.imshow(bbx_map)
    plt.axis("off")
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved {overlay_path}!")
    return bbx_map



def run_TC_one_patch(crop_norm, TC):
    input_img = np.expand_dims(np.array(crop_norm), axis=0)
    input_img = tf.keras.applications.mobilenet.preprocess_input(input_img)
    TC_epi, TC_str, TC_adi = TC.predict(input_img)[0]
    return TC_epi, TC_str, TC_adi



## make predictions for patch images
class PatchDataset(Sequence):
    def __init__(self, dataframe, patch_size, batch_size=32, shuffle=False):
        self.dataframe = dataframe
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_paths = self.dataframe['file_path'].values  # File paths
        self.labels = self.dataframe['class'].values  # Labels
        self.on_epoch_end()  # Shuffle data initially
    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images, valid_labels, valid_paths = self.__data_generation(batch_paths, batch_labels)
        return batch_images, valid_labels, valid_paths  # Return valid file paths and their labels
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.image_paths, self.labels))
            np.random.shuffle(temp)
            self.image_paths, self.labels = zip(*temp)
    def __data_generation(self, batch_paths, batch_labels):
        batch_images = []
        valid_labels = []  # Store labels for valid images
        valid_paths = []  # Store paths of successfully loaded images
        for img_path, label in zip(batch_paths, batch_labels):
            try:
                img = Image.open(img_path)
                img = Reinhard(np.array(img))  # Apply Reinhard stain normalization
                img = Image.fromarray(img)  # Convert back to PIL for resizing
                img = img.resize((self.patch_size, self.patch_size))
                img = np.array(img, dtype=np.float32)  # Convert to NumPy array
                img = preprocess_input(img)  # Apply MobileNetV2 preprocessing
                batch_images.append(img)
                valid_labels.append(label)  # Append corresponding label
                valid_paths.append(img_path)  # Only keep successfully loaded file paths
            except (OSError, Image.DecompressionBombError) as e:
                print(f"Skipping corrupted image: {img_path} ({str(e)})")
        
        return np.array(batch_images), np.array(valid_labels), valid_paths  # Return valid batch images, labels, and paths
