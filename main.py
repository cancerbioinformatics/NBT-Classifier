from run_TC import run_TC
from run_bbx import run_bbx
import argparse

parser = argparse.ArgumentParser(description="Run Tissue-Classifier to get a 3-class mask")

# Boolean flags
parser.add_argument("--save_bbxpng", action='store_true', help="Localize peri-lobular regions and save a .png file")
parser.add_argument("--save_patchcsv", action='store_true', help="Localize peri-lobular regions and save a .csv file storing patches classes and coordinates")
parser.add_argument("--save_TCmap", action='store_true', help="Save .png file of the predicted Tissue type mask")
parser.add_argument("--free_space", action='store_true', help="Remove the intermediate large files")

# Required paths
parser.add_argument("--WSI", type=str, required=True, help="The folder storing whole slide images")
parser.add_argument("--MASK", type=str, required=True, help="The folder storing foreground masks")
parser.add_argument("--TC_output", type=str, required=True, help="The folder containing output files from running Tissue-Classifier")

# Defaults
parser.add_argument("--WEIGHT", type=str, default='./data/NBT_512px.h5', help="Path to the weights for Tissue-Classifier")
parser.add_argument("--foreground_thes", type=float, default=0.7, help="Only process patches with tissue area over this threshold")
parser.add_argument("--patch_size", type=int, default=128, help="Fixed patch size (µm)")
parser.add_argument("--upsample", type=int, default=32, help="Upsample the tissue type mask {upsample} times")
parser.add_argument("--small_objects", type=int, default=400000, help="Remove small objects/holes below this size (at 40x)")
parser.add_argument("--roi_width", type=int, default=250, help="Width (µm) of the peri-lobular regions to include")

args = parser.parse_args()

if __name__ == "__main__":
    run_TC(args)  
    run_bbx(args)









        
