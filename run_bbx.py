import os
os.chdir("/scratch/users/k21066795/prj_normal/awesome_normal_breast/scripts")
import glob
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from utils import parse_patch_size,process_TCmask,bbx_overlay,get_roi_ids,roi_id2patch_id,save_patchcsv



def run_bbx(args):
    TCoutputs = os.listdir(args.TC_output)
    random.shuffle(TCoutputs)
    
    for wsi_id in TCoutputs:
        print(wsi_id)
        output_dir=f"{args.TC_output}/{wsi_id}"

        TC_maskpt = glob.glob(f"{output_dir}/{wsi_id}_TC*mask.npy")
        if TC_maskpt:
            TC_maskpt = TC_maskpt[0]
            
            wsi_pt = glob.glob(f"{args.WSI}/{wsi_id}*.*")[0] 
            try:
                wsi = openslide.OpenSlide(wsi_pt)
            except Exception as e:
                print(f"Error opening WSI {wsi_pt}: {e}")
                continue
        
            patch_size, _ = parse_patch_size(wsi, patch_size=args.patch_size)
            epi_mask, roi_width, wsi_mask_ratio = process_TCmask(wsi_pt, TC_maskpt, args.upsample, args.small_objects, args.roi_width) 

            if args.save_bbxpng:
                overlay_pt = f"{output_dir}/{wsi_id}_bbx.png"
                if not os.path.exists(overlay_pt): 
                    bbx_overlay(epi_mask, overlay_pt, roi_width)
                    print(f"{overlay_pt} saved!")
    
            if args.save_patchcsv:
                patch_csv = f"{output_dir}/{os.path.basename(output_dir)}_patch.csv"
                if not os.path.exists(patch_csv):
                    roi_ids = get_roi_ids(epi_mask, wsi_id, roi_width, args.upsample, wsi_mask_ratio)
                    save_patchcsv(roi_ids, patch_size, TC_maskpt, output_dir)
            



parser = argparse.ArgumentParser(description="Run Tissue-Classifier to get a 3-class mask")
parser.add_argument("--WSI", type=str, help="the folder storing foreground masks")
parser.add_argument("--TC_output", type=str, help="the folder containing output files from running Tissue-Classifier")
parser.add_argument("--patch_size", type=int, default=128, help="the fix patch size (um)")
parser.add_argument("--upsample", type=int, default=32, help="upsample the tissue type mask {upsample} times")
parser.add_argument("--small_objects", type=int, default=400000, help="small objects/holes under this pixel size (at 40x magnification) will be removed or filled, for example, 400000 pixels at 40x are approximately 1.5 patches of 512x512 pixels; use 50000 for 20x")
parser.add_argument("--roi_width", type=int, default=250, help="the width (um) of the peri-lobular regions to include")
parser.add_argument("--save_bbxpng", action='store_true', help="whether localise perilobular regions and save a .png file")
parser.add_argument("--save_patchcsv", action='store_true', help="whether localise perilobular regions and save a .csv file storing patches classes and coordinates")


args = parser.parse_args()
run_bbx(args)
print("Localising ROIs and tessellation finished!")
