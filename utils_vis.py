import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
import cv2
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image
import openslide
from typing import Tuple
import staintools

# Tensorflow Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Resizing
from tensorflow.keras.utils import Sequence



def Reinhard(img_arr, standard_img="/scratch_tmp/users/k21066795/he_shg_synth_workflow/thumbnails/he.jpg"):
    target = staintools.read_image(standard_img)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target)
    img_to_transform = staintools.LuminosityStandardizer.standardize(img_arr)
    return normalizer.transform(img_to_transform)



def plot_oneline(img_list, caption_list, figure_size, save_pt=None):
    fig, axes = plt.subplots(1, len(img_list), figsize=figure_size)
 
    for index in range(0, len(img_list)):
            axes[index].imshow(img_list[index])
            axes[index].axis("off")
            caption_i = caption_list[index]
            
            if isinstance(caption_i, str):
                axes[index].set_title(f"{caption_i}")
            else:
                axes[index].set_title(f"{np.around(caption_i, 2)}")
    if save_pt is not None:
        plt.savefig(save_pt,pad_inches=0, bbox_inches="tight", dpi=300)



def plot_multiple(img_list, caption_list, grid_x, grid_y, figure_size, cmap="gray", save_pt=None):
    if len(img_list) != len(caption_list):
        raise ValueError("The number of images must match the number of captions.")
    
    if len(img_list) < grid_x * grid_y:
        raise ValueError("Not enough images to fill the specified grid dimensions.")

    fig, axes = plt.subplots(grid_x, grid_y, figsize=figure_size)

    # Flatten axes if grid is more than 1x1
    axes = axes.ravel() if grid_x * grid_y > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < len(img_list):
            ax.imshow(img_list[i], cmap=cmap)
            ax.axis("off")
            ax.set_title(caption_list[i])
        else:
            ax.axis("off")  # Hide extra axes if images are fewer than grid slots

    plt.tight_layout()

    if save_pt:
        plt.suptitle(os.path.splitext(os.path.basename(save_pt))[0])  # Set figure title
        plt.savefig(save_pt, pad_inches=0, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_pt}")

    plt.show()



def get_GAPCAM(img_path, model, layer='conv_pw_13_relu', label_index=None, with_grids=True, patch_size=None):
    img = Image.open(img_path)
    img = np.array(img.resize((patch_size, patch_size)))
    img_norm = Reinhard(img)  # Ensure Reinhard normalization is implemented
    input_img = np.expand_dims(img_norm, axis=0)
    input_img = tf.keras.applications.mobilenet.preprocess_input(input_img)

    model_CAM = Model([model.inputs], 
                      [model.get_layer(layer).output, 
                       model.output])

    activation_maps, predictions = model_CAM(input_img)
    pred_prob = np.around(predictions[0], 2)

    if label_index is None:
        label_index = np.argmax(predictions)

    activation_maps = np.squeeze(activation_maps.numpy())  # Shape (h, w, channels)
    weights_pred = model.layers[-1].get_weights()[0][:, label_index]
    weights_dense2 = model.layers[-2].get_weights()[0]
    weights_dense1 = model.layers[-4].get_weights()[0]

    heatmap_weights = tf.tensordot(weights_dense2, weights_pred, axes=1)
    heatmap_weights = tf.tensordot(weights_dense1, heatmap_weights, axes=1)

    weighted_activation_maps = activation_maps.copy()
    for i in range(weighted_activation_maps.shape[-1]):
        weighted_activation_maps[:, :, i] *= heatmap_weights[i]

    CAM = np.mean(weighted_activation_maps, axis=-1)  # Raw CAM
    CAM = np.maximum(CAM, 0)  # ReLU to remove negative values
    CAM /= CAM.max()  # Normalize to [0,1]

    # Create heatmap (512x512)
    cam_heatmap = cv2.resize(CAM, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR) # cv2.INTER_NEAREST, cv2.INTER_LINEAR
    cam_heatmap = (cam_heatmap * 255).astype(np.uint8)
    cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_VIRIDIS)

    # Superimpose CAM on original image
    super_imposed_cam = cv2.addWeighted(img.astype("float32"), 0.5, cam_heatmap.astype("float32"), 0.5, 0.0)
    super_imposed_cam = super_imposed_cam.astype(np.uint8)

    if with_grids:
        # Draw 16x16 grid on super_imposed_cam
        step = patch_size // CAM.shape[0]  # Grid cell size
        for i in range(1, CAM.shape[0]):
            # Draw horizontal lines
            cv2.line(super_imposed_cam, (0, i * step), (patch_size, i * step), (255, 255, 255), 1)
            # Draw vertical lines
            cv2.line(super_imposed_cam, (i * step, 0), (i * step, patch_size), (255, 255, 255), 1)

    return pred_prob, img, CAM, cam_heatmap, super_imposed_cam



def get_gradCAM(img_path, model, layer='conv_pw_13_relu', label_index=None, with_grids=True, patch_size=None):
    img = Image.open(img_path)
    img = np.array(img.resize((patch_size, patch_size)))
    img_norm = Reinhard(img)
    input_img = np.expand_dims(img_norm, axis=0)
    input_img = tf.keras.applications.mobilenet.preprocess_input(input_img)


    gradCAM_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer).output, model.output]
    )

    # compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradCAM_model(input_img)
        if label_index is None:
            label_index = tf.argmax(preds[0])
        class_channel = preds[:, label_index]
    print(preds)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    CAM = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    CAM = tf.squeeze(CAM)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    CAM = tf.maximum(CAM, 0) / tf.math.reduce_max(CAM)
    CAM = np.array(CAM)
    
    cam_heatmap = cv2.resize(CAM, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_LINEAR)
    cam_heatmap = cam_heatmap *255
    cam_heatmap = np.clip(cam_heatmap, 0, 255).astype(np.uint8)
    cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_VIRIDIS)
    print(cam_heatmap.shape)
    
    super_imposed_cam = cv2.addWeighted(img.astype("float32"), 0.5, cam_heatmap.astype("float32"), 0.5, 0.0)
    super_imposed_cam = super_imposed_cam.astype(np.uint8)

    if with_grids:
        # Draw 16x16 grid on super_imposed_cam
        step = patch_size // CAM.shape[0]  # Grid cell size
        for i in range(1, CAM.shape[0]):
            # Draw horizontal lines
            cv2.line(super_imposed_cam, (0, i * step), (patch_size, i * step), (255, 255, 255), 1)
            # Draw vertical lines
            cv2.line(super_imposed_cam, (i * step, 0), (i * step, patch_size), (255, 255, 255), 1)

    return preds, img, CAM, cam_heatmap, super_imposed_cam



def generate_annotationMask(wsi_path, anno_pt, level=None):
    slide = openslide.OpenSlide(wsi_path)
    with open(anno_pt, "r") as f:
        shapes = json.load(f)

    if level is None:
        level = slide.level_count - 1
    scale_factor = 1 / slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]

    mask = np.zeros((height, width, 3), dtype=np.uint8)

    color_map = {
        "epithelials": (255, 0, 0),  # Red for epithelium
        "stroma": (0, 255, 0),      # Green for stroma
        "miscellaneous": (0, 0, 255)   # Blue for adipocytes
    }

    for shape in shapes.get("features", []):
        try:
            cls = shape["properties"]["classification"]["name"]
            if cls not in color_map:
                continue  # Ignore unrecognized classifications

            points = np.array(shape["geometry"]["coordinates"][0]) * scale_factor
            points = points.astype(int)

            # Draw filled polygon and contour
            cv2.fillPoly(mask, [points], color_map[cls])
            cv2.drawContours(mask, [points], -1, color_map[cls], thickness=1)
        
        except (KeyError, ValueError, TypeError) as e:
            print(f"Skipping shape due to error: {e}")

    return mask




def patch_overlay_wsi(patch_df, wsi_pt, patch_size, figsize=(10, 10)):    
    wsi = openslide.OpenSlide(wsi_pt)
    level_max = wsi.level_count - 1
    wsi_level = np.array(wsi.read_region((0,0), level_max, wsi.level_dimensions[level_max]).convert("RGB"))
    down_ratio = wsi.level_downsamples[level_max]
    
    patch_map = np.full((wsi_level.shape[0], wsi_level.shape[1]), np.nan)    
    patch_id_list = list(patch_df["patch_id"])
    cls_list = list(patch_df["cls"])
    
    for patch_id, cls_i in zip(patch_id_list, cls_list):
        grid_x, grid_y, patch_size = patch_id.split("_")[-3:]
        grid_x, grid_y, patch_size = int(grid_x), int(grid_y), int(patch_size)
        
        x1 = int(grid_x * patch_size // down_ratio)
        y1 = int(grid_y * patch_size // down_ratio)
        
        x2 = np.min([patch_map.shape[0], int(x1+patch_size//down_ratio)])
        y2 = np.min([patch_map.shape[1], int(y1+patch_size//down_ratio)])
        patch_map[y1:y2, x1:x2] = int(cls_i)  # 0,1,2,3
    
    plt.figure(figsize=figsize) 
    color = ["Red", "green", "blue"] 
    sns.set_palette(color)
    
    heatmap = sns.heatmap(patch_map, cmap = sns.color_palette(), alpha = 0.5, 
                          xticklabels=False, yticklabels=False)
    heatmap.imshow(wsi_level,
                   aspect = heatmap.get_aspect(),
                   extent = heatmap.get_xlim() + heatmap.get_ylim(),
                   zorder = 0)
    
    return heatmap.get_figure()




def parse_roi_id(roi_id):
    wsi_id = "_".join(roi_id.split("_")[:-4])
    x,y,w,h = int(roi_id.split("_")[-4]), int(roi_id.split("_")[-3]), int(roi_id.split("_")[-2]), int(roi_id.split("_")[-1])
    return wsi_id, x, y, w, h



def show_roi(wsi, roi_id):
    wsi_id, x_orig, y_orig, wid_orig, heigh_orig = parse_roi_id(roi_id)
    roi_img = wsi.read_region((int(x_orig), int(y_orig)), 0, (int(wid_orig), int(heigh_orig))).convert("RGB")
    im = Image.fromarray(np.array(roi_img))
    return im
    


def parse_patch_id(patch_id):
    _,_,_,_,grid_x,grid_y,patch_size = patch_id.split('_')[-7:]
    orig_x = int(int(grid_x) * int(patch_size))
    orig_y = int(int(grid_y) * int(patch_size))
    return orig_x, orig_y, patch_size



def show_patch(wsi, patch_id, save_pt = None):
    orig_x, orig_y, patch_size = parse_patch_id(patch_id)
    img = wsi.read_region((int(orig_x), int(orig_y)), 0, (int(patch_size), int(patch_size))).convert("RGB")
    plt.imshow(img)
    plt.axis('off')

    if save_pt is not None:
        Image.fromarray(np.array(img)).save(save_pt)
    return img

