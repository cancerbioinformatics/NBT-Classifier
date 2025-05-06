import argparse
import glob
import os
from PIL import Image
import numpy as np
import openslide
from utils import *

import warnings
warnings.filterwarnings('ignore', 
    message='.*Attempting to register factory for plugin.*')
warnings.filterwarnings('ignore',
    message='.*computation placer already registered.*')


def parse_args():
    parser = argparse.ArgumentParser(description="Process whole slide images (WSIs) and predict tissue classes.")

    # Arguments for input directories
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder containing mask images.")
    parser.add_argument('--wsi_folder', type=str, required=True, help="Path to the folder containing WSIs.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to save output results.")

    # Arguments for model
    parser.add_argument('--model_type', type=str, required=True, choices=['TC_512', 'TC_1024'], help="Specify the model: 'TC_512' or 'TC_1024'.")

    # Arguments for processing
    parser.add_argument('--patch_size_microns', type=int, choices=[128, 256], default=128, help="Patch size for tissue classification. Must be one of [128, 256].")
    parser.add_argument('--foreground_threshold', type=float, default=0.7, help="Threshold for foreground mask.")
    parser.add_argument('--free_space', action='store_true', help="Whether to free up space by deleting temporary files.", default=False)
    parser.add_argument('--upscale_factor', type=int, default=32, help="Upscale factor for processing masks.")
    parser.add_argument('--small_objects', type=int, choices=[400000, 250000], default=400000, help="Minimum object size to retain in the mask. Must be one of [400000, 250000].")
    parser.add_argument('--roi_width', type=int, default=250, help="Width of ROI for localized processing (None for all patches).")
    
    parser.add_argument('--use_multithreading', action='store_true', help="Enable multithreaded tile processing.")
    parser.add_argument('--max_workers', type=int, default=8, help="Number of worker threads/processes to use if multithreading is enabled.")

    return parser.parse_args()




def process_all_slides(mask_folder, wsi_folder, output_folder, model_type, 
                       patch_size_microns=128, foreground_thes=0.7, free_space=False, use_multithreading=True, max_workers=8,
                       upscale_factor=32, small_objects=400000, roi_width=250):

    if model_type == 'TC_512':
        IMAGE_SIZE = (512, 512)
        model_weights = './data/TC_512px.h5'
        model = get_TC(model_weights, image_size=IMAGE_SIZE, num_classes=3, output_features=False)
    elif model_type == 'TC_1024':
        IMAGE_SIZE = (1024, 1024)
        model_weights = './data/TC_1024px.h5'
        model = get_TC(model_weights, image_size=IMAGE_SIZE, num_classes=3, output_features=False)
    
    foreground_masks = glob.glob(f"{mask_folder}/*/*_mask_use.png")
    for mask_path in foreground_masks:
        wsi_name = get_wsiname(mask_path)
        wsi_paths = glob.glob(f"{wsi_folder}/{wsi_name}*.*")
        if not wsi_paths:
            print(f"No WSI found for {wsi_name}, skipping...")
            continue
        
        wsi_path = wsi_paths[0]
        require_bounds = ".mrxs" in wsi_path
        try:
            wsi = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"Openslide cannot open {wsi_path}: {e}")
            return None  # Return None to indicate failure
    
        print(f"Processing {wsi_name}...")
        output_dir = os.path.join(output_folder, wsi_name)
        os.makedirs(output_dir, exist_ok=True)
        patch_size, _ = parse_patch_size(wsi, patch_size_microns)

        TC_maskpt = os.path.join(output_dir, f"{wsi_name}_{model_type}_probmask.npy")
        tissue_map = run_TC_one_slide(wsi, mask_path, TC_maskpt, patch_size, foreground_thes, IMAGE_SIZE, model, free_space, use_multithreading, max_workers)

        Allpatch_pt = f"{output_dir}/{wsi_name}_{model_type}_patch_all.csv"
        cls_df = save_Allpatch(tissue_map, patch_size, Allpatch_pt) 
        
        json_pt = f"{output_dir}/{wsi_name}_{model_type}_cls_wsi.json"
        get_JSON(cls_df, json_pt, patch_size, require_bounds)

        tc_map_path = os.path.join(output_dir, f"{wsi_name}_{model_type}.png")
        save_tc_map(tissue_map, wsi_path, tc_map_path)

        epi_mask, roi_width, wsi_mask_ratio = process_TCmask(wsi, tissue_map, upscale_factor, small_objects, roi_width)
        epi_mask_pt = f"{output_dir}/{wsi_name}_{model_type}_epi_({int(wsi_mask_ratio)},0,0,{epi_mask.shape[1]},{epi_mask.shape[0]})-mask.png"
        save_epi_mask(epi_mask, epi_mask_pt)

        patch_pt = f"{output_dir}/{wsi_name}_{model_type}_patch_roi.csv"
        patch_df = save_ROIpatch(tissue_map, epi_mask, wsi_mask_ratio, roi_width, patch_size, patch_pt)
        
        json_pt = f"{output_dir}/{wsi_name}_{model_type}_cls_roi.json"
        get_JSON(patch_df, json_pt, patch_size, require_bounds)

        overlay_path = f"{output_dir}/{wsi_name}_{model_type}_bbx.png"
        bbx_map = bbx_overlay(epi_mask, overlay_path, roi_width)




def main():
    args = parse_args()

    process_all_slides(mask_folder=args.mask_folder, wsi_folder=args.wsi_folder, output_folder=args.output_folder,
                       model_type=args.model_type, patch_size_microns=args.patch_size_microns, foreground_thes=args.foreground_threshold,
                       free_space=args.free_space, use_multithreading=args.use_multithreading, max_workers=args.max_workers,
                       upscale_factor=args.upscale_factor, small_objects=args.small_objects, roi_width=args.roi_width)




if __name__ == "__main__":
    main()



