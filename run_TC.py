import os
import glob
import argparse
import numpy as np
import time
import random
from PIL import Image
import openslide
from utils import get_wsiname, SlideIterator, get_TC, WsiNpySequence, parse_patch_size, get_TC_predictions



def run_TC(args):
    foreground_masks = glob.glob(f"{args.MASK}/*/*_mask_use.png")
    # foreground_masks = [i for i in foreground_masks if " HE" not in i]
    # foreground_masks = [i for i in foreground_masks if "_FPE_" not in i]
    random.shuffle(foreground_masks)
    
    for mask_pt in foreground_masks:
        wsiname = get_wsiname(mask_pt)
        # wsiname = os.path.basename(mask_pt).split(".svs")[0]
        print(f"preprocessing {wsiname}")
        
        output_dir=f"{args.TC_output}/{wsiname}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        if not os.path.exists(f"{output_dir}/{wsiname}_TCprobmask.npy"):
            mask_arr = np.array(Image.open(mask_pt))
            mask_path = mask_pt.replace(".png", ".npy")
            np.save(mask_path, (mask_arr/255).astype("uint8"))
            wsi_pt = glob.glob(f"{args.WSI}/{wsiname}*.*")[0]
            print(wsi_pt)
        
            try:
                # tissue_map, image_shape = get_TC_predictions(wsi_pt, mask_path, args)
                wsi=openslide.OpenSlide(wsi_pt)
                print(f"vectorise {wsiname}")
                        
                output_pattern = os.path.join(output_dir, f"{wsiname}_pattern")
                time1 = time.time()
                si = SlideIterator(wsi=wsi, image_level=0, mask_path=mask_path, threshold_mask=args.foreground_thes)
                
                patch_size, _ = parse_patch_size(wsi, args.patch_size)
                        
                si.save_array(patch_size=patch_size, stride=patch_size, output_pattern=output_pattern, downsample=1)
                print(f"{time.time() - time1} seconds to vectorise {wsiname}!")
                        
                print(f"Tissue-Classifier is predicting {wsiname}")
                tissue_classifier=get_TC(args.WEIGHT)
                wsi_sequence = WsiNpySequence(wsi_pattern=output_pattern, batch_size=8)
                tc_predictions = tissue_classifier.predict_generator(generator=wsi_sequence, steps=len(wsi_sequence),verbose=1)
                
                xs = wsi_sequence.xs
                ys = wsi_sequence.ys
                image_shape = wsi_sequence.image_shape
                tissue_map = np.ones((image_shape[1], image_shape[0], tc_predictions.shape[1])) * np.nan
            
                for patch_feature, x, y in zip(tc_predictions, xs, ys):
                    tissue_map[y, x, :] = patch_feature

        
                tissue_map[np.isnan(tissue_map)] = 0
                npy_pt = f"{output_dir}/{wsiname}_TCprobmask.npy"
                np.save(npy_pt, tissue_map)
                print(f"{npy_pt} saved!")
    
                # save tissue classification heatmap
                if args.save_TCmap:
                    if ".mrxs" in os.path.basename(wsi_pt):
                        bounds_h = int(wsi.properties['openslide.bounds-height'])//patch_size
                        bounds_w = int(wsi.properties['openslide.bounds-width'])//patch_size
                        bounds_x = int(wsi.properties['openslide.bounds-x'])//patch_size
                        bounds_y = int(wsi.properties['openslide.bounds-y'])//patch_size
                        tissue_map = tissue_map[bounds_y:(bounds_y+bounds_h), bounds_x:(bounds_x+bounds_w),:]
                
                    im = Image.fromarray((tissue_map * 255).astype("uint8"))
                    im.save(f"{output_dir}/{wsiname}_TC_({patch_size},0,0,{image_shape[0]},{image_shape[1]}).png")
                
                if args.free_space:
                    for i in glob.glob(f"{output_dir}/*pattern*"):
                        os.remove(i)
                        print(f"{i} removed!")

            except:
                print(f"Openslide can not open {wsi_pt}")
                continue




parser = argparse.ArgumentParser(description="Run Tissue-Classifier to get a 3-class mask")
parser.add_argument("--save_TCmap", type=bool, default=True, help="Save .png file of the predicted Tissue type mask")
parser.add_argument("--free_space", type=bool, default=True, help="Remove the intermediate large files")
parser.add_argument("--WSI", type=str, help="the folder storing foreground masks")
parser.add_argument("--MASK", type=str, help="the folder storing foreground masks")
parser.add_argument("--TC_output", type=str, help="the folder containing output files from running Tissue-Classifier")
parser.add_argument("--WEIGHT", type=str, default='data/MobileNet512px.h5', help="the path of the weights for Tissue-Classifier")
parser.add_argument("--foreground_thes", type=float, default=0.7, help="Only process patches with tissue area over this threshold")
parser.add_argument("--patch_size", type=int, default=128, help="the fix patch size (um)")



args = parser.parse_args()
run_TC(args)


