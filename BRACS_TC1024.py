import glob
import numpy as np
import staintools
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Resizing
from tensorflow.keras.utils import Sequence

import pandas as pd
from pathlib import Path
from tqdm import tqdm  
import numpy as np
from PIL import Image
import random
import os

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm
        

def crop_patch(image, x, y, patch_size=1024):
    return image.crop((x, y, x + patch_size, y + patch_size))


def Reinhard(img_arr, standard_img="/scratch/users/k21066795/NBT-Classifier/data/he.jpg"):
    target = staintools.read_image(standard_img)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target)
    img_to_transform = staintools.LuminosityStandardizer.standardize(img_arr)
    return normalizer.transform(img_to_transform)


def get_TC1024(weights, output_features=False):
    IMAGE_SIZE = (1024, 1024)
    NUM_CLASSES = 3
    initializer = tf.keras.initializers.GlorotNormal()
    # Define the base MobileNet model
    net = MobileNet(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='Dense_1', kernel_initializer=initializer)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='Dense_2', kernel_initializer=initializer)(x)
    feature_layer = x  # Save this layer for feature extraction
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='Predictions')(x)
    # Create the model
    if output_features:
        # Create the model with two outputs: predictions and features
        net_final = Model(inputs=net.input, outputs=[output_layer, feature_layer], trainable=False)
    else:
        # Create the model with only predictions
        net_final = Model(inputs=net.input, outputs=output_layer, trainable=False)
    # Load the weights
    net_final.load_weights(weights)
    return net_final


def run_TC_one_patch(crop_norm, TC):
    input_img = np.expand_dims(np.array(crop_norm), axis=0)
    input_img = tf.keras.applications.mobilenet.preprocess_input(input_img)
    preds, features = TC.predict(input_img)  # 先得到 features 和预测
    TC_epi, TC_str, TC_adi = preds[0]         # preds[0] 是三分类概率
    return TC_epi, TC_str, TC_adi, features[0]  # features[0] 是 512-d 向量


def main(batch_index):
    # Configuration
    basefolder = "/scratch/prj/cb_microbiome/recovered/Siyuan/prj_NBTClassifier/npj_revision"
    image_folder = '/scratch/prj/cb_normalbreast/prj_BreastAgeNet/WSIs/BRACS_ROI'

    # Load and split data by batch
    sampled_df = pd.read_csv(f"{basefolder}/BRACS_patch_ids_CenterSampling_sampled8000.csv")
    
    # Calculate batch range
    start_idx = batch_index * 28000
    end_idx = min((batch_index + 1) * 28000, len(sampled_df))
    batch_df = sampled_df.iloc[start_idx:end_idx]
    
    print(f"Processing batch {batch_index}: indices {start_idx}-{end_idx}, total {len(batch_df)} patches")

    # Load model
    weights = '/scratch/prj/cb_histology_data/Siyuan/Docker_test/nbtclassifier/NBT-Classifier/data/TC_1024px.h5'
    TC1024 = get_TC1024(weights, output_features=True)
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Using device: {device}")

    # Set output path
    save_path = Path(f"{basefolder}/BRACS_patch_ids_overlap25_sampled3000_TC1024_batch{batch_index}.csv")

    # Check existing results
    if save_path.exists():
        existing_ids = set(pd.read_csv(save_path, usecols=["patch_id"])["patch_id"].astype(str))
        print(f"Found existing file with {len(existing_ids)} processed patches")
    else:
        existing_ids = set()
        print("Creating new output file")

    # Process current batch
    buffer = []
    batch_size = 100
    
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Processing batch {batch_index}"):
        patch_id = str(row['patch_id'])
        
        # Skip already processed patches
        if patch_id in existing_ids:
            continue
        
        roi_id = row['roi_id']
        x, y = int(row['x']), int(row['y'])

        # Find ROI image
        image_paths = glob.glob(f"{image_folder}/*/*/{roi_id}.png")
        if not image_paths:
            print(f"ROI image not found for {roi_id}")
            continue
        image_path = image_paths[0]

        # Extract and preprocess patch
        image = Image.open(image_path)
        patch_image = crop_patch(image, x, y, patch_size=1024)
        crop_norm = Image.fromarray(Reinhard(np.array(patch_image)))
        
        # Run tissue classifier
        TC_epi, TC_str, TC_adi, features = run_TC_one_patch(crop_norm, TC1024)
        cls_index = np.argmax([TC_epi, TC_str, TC_adi])
        cls_label = ['epithelium', 'stroma', 'adipocytes'][cls_index]

        # Format features
        feature_dict = {f"feature{i}": features[i] for i in range(len(features))}
        
        # Create result row
        row_dict = {
            "patch_id": patch_id,
            "cls": cls_label,
            "TC_epi": TC_epi,
            "TC_str": TC_str,
            "TC_adi": TC_adi,
            "source": row['source'],  # Original class from dataset
            **feature_dict
        }

        buffer.append(row_dict)

        # Save buffer when full
        if len(buffer) >= batch_size:
            pd.DataFrame(buffer).to_csv(
                save_path,
                mode="a",
                header=not save_path.exists(),
                index=False
            )
            existing_ids.update([d["patch_id"] for d in buffer])
            buffer = []

    # Save remaining buffer
    if buffer:
        pd.DataFrame(buffer).to_csv(
            save_path,
            mode="a",
            header=not save_path.exists(),
            index=False
        )
        existing_ids.update([d["patch_id"] for d in buffer])

    print(f"Batch {batch_index} completed! Processed {len(batch_df)} patches")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patches in batches using tissue classifier")
    parser.add_argument("--batch_index", type=int, required=True, help="Batch index (0, 1, 2, ...)")
    args = parser.parse_args()
    
    main(args.batch_index)






    