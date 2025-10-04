import os
import csv
import pandas as pd
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm  
from pathlib import Path
import numpy as np
import staintools
import pandas as pd
from PIL import Image

import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader



def get_features_and_output(model, x):
    model.eval()
    with torch.no_grad():  # Disable gradient tracking for inference
        # Add batch dimension
        x = x.unsqueeze(0)  # Shape: [1, channels, height, width]
        
        # Pass through the network up to the penultimate layer
        x = model.conv1(x)  # First convolution layer
        x = model.bn1(x)    # BatchNorm layer
        x = model.relu(x)   # ReLU activation
        x = model.maxpool(x)  # Max pooling
        
        # Pass through the rest of the network layers
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        
        # Extract features from the avgpool layer
        features = model.avgpool(x)  # Shape: [batch_size, 512, 1, 1]
        
        # Flatten features to a 1D vector (for each image)
        features = torch.flatten(features, 1)  # Shape: [batch_size, 512]
        
        # Obtain the final model output (prediction)
        output = model.fc(features)  # Shape: [batch_size, num_classes]

    return features, output



def crop_patch(image, x, y, patch_size=1024):
    return image.crop((x, y, x + patch_size, y + patch_size))


def Reinhard(img_arr, standard_img="/scratch/users/k21066795/NBT-Classifier/data/he.jpg"):
    target = staintools.read_image(standard_img)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target)
    img_to_transform = staintools.LuminosityStandardizer.standardize(img_arr)
    return normalizer.transform(img_to_transform)




import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import glob
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import csv
import os
from tqdm import tqdm

def main(batch_index):
    # Configuration
    basefolder = "/scratch/prj/cb_microbiome/recovered/Siyuan/prj_NBTClassifier/npj_revision"
    image_folder = '/scratch/prj/cb_normalbreast/prj_BreastAgeNet/WSIs/BRACS_ROI'

    # Load and split data by batch
    sampled_df = pd.read_csv(f"{basefolder}/BRACS_patch_ids_CenterSampling_sampled8000.csv")
    
    # Calculate batch range
    start_idx = batch_index * 14000
    end_idx = min((batch_index + 1) * 14000, len(sampled_df))
    batch_df = sampled_df.iloc[start_idx:end_idx]
    
    print(f"Processing batch {batch_index}: indices {start_idx}-{end_idx}, total {len(batch_df)} patches")

    # Load model
    histoROI = "/scratch/prj/cb_normalbreast/prj_BreastAgeNet/CKPTs/histoROI_weights.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model6 = models.resnet18().to(device)
    model6.fc = torch.nn.Linear(512, 6).to(device)
    model6.load_state_dict(torch.load(histoROI, map_location=device))
    model6.eval()

    # Set output path
    csv_file = Path(f"{basefolder}/BRACS_patch_ids_overlap25_sampled3000_histoROI_batch{batch_index}.csv")

    # Define header
    header = [
        "patch_id", "cls", "Epithelial", "Stroma", "Adipose", "Artefact", 
        "Miscellaneous", "Lymphocytes", "source"
    ] + [f"feature{i}" for i in range(512)]

    # Check if file exists and get processed patch IDs
    file_exists = csv_file.exists()
    processed_patch_ids = set()
    
    if file_exists:
        try:
            existing_df = pd.read_csv(csv_file, usecols=["patch_id"])
            processed_patch_ids = set(existing_df["patch_id"].astype(str))
            print(f"Found existing file with {len(processed_patch_ids)} processed patches")
        except:
            print("Existing file found but could not read patch_ids")
            file_exists = False

    # Image transformation
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    # Softmax for probability conversion
    sm = torch.nn.Softmax(dim=1)
    class_names = ['Epithelial', 'Stroma', 'Adipose', 'Artefact', 'Miscellaneous', 'Lymphocytes']

    # Process current batch
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Processing batch {batch_index}"):
        patch_id = str(row['patch_id'])
        
        # Skip already processed patches
        if patch_id in processed_patch_ids:
            continue
        
        roi_id = row['roi_id']
        x, y = int(row['x']), int(row['y'])
        
        # Find ROI image
        image_paths = glob.glob(f"{image_folder}/*/*/{roi_id}.png")
        if not image_paths:
            print(f"ROI image not found for {roi_id}")
            continue
        image_path = image_paths[0]
        
        try:
            # Load and process image
            image = Image.open(image_path)
            patch_image = crop_patch(image, x, y)
            crop_norm = Image.fromarray(Reinhard(np.array(patch_image)))

    
            # Transform image
            img = base_transform(crop_norm).to(device)  # Add batch dimension
            
            # Get model predictions
            features, out = get_features_and_output(model6, img)
            
            # Apply softmax to get probabilities
            probs = sm(out)
            probs = probs.cpu().detach().numpy()[0]
            
            # Get predicted class
            pred_idx = np.argmax(probs)
            cls_label = class_names[pred_idx]
            
            # Process features
            features_np = features.cpu().detach().numpy()
            if features_np.ndim > 1:
                features_np = features_np.flatten()
            
            feature_dict = {f"feature{i}": features_np[i] for i in range(len(features_np))}
            
            # Extract individual probabilities
            Epithelial_prob = probs[0]
            Stroma_prob = probs[1]
            Adipose_prob = probs[2]
            Artefact_prob = probs[3]
            Miscellaneous_prob = probs[4]
            Lymphocytes_prob = probs[5]
            
            # Prepare row data
            row_data = {
                "patch_id": patch_id,
                "cls": cls_label,
                "Epithelial": Epithelial_prob,
                "Stroma": Stroma_prob,
                "Adipose": Adipose_prob,
                "Artefact": Artefact_prob,
                "Miscellaneous": Miscellaneous_prob,
                "Lymphocytes": Lymphocytes_prob,
                "source": row['source'],
                **feature_dict
            }
            
            # Write to CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerow(row_data)
            
            # Add to processed set
            processed_patch_ids.add(patch_id)
            
        except Exception as e:
            print(f"Error processing patch {patch_id}: {e}")
            continue

    print(f"Batch {batch_index} completed! Processed {len(batch_df)} patches")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patches in batches using HistoROI model")
    parser.add_argument("--batch_index", type=int, required=True, help="Batch index (0, 1, 2, ...)")
    args = parser.parse_args()
    
    main(args.batch_index)





