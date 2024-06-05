#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Mon May 20 18:12:53 2024
@author: marcalbesa

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure
from PIL import Image

# Function to preprocess and save images
def preprocess_and_save(images_list, conditions_list, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i, cond in enumerate(conditions_list):
        cond_path = os.path.join(output_path, f"ML_A549_{cond}_PhaD_20X_RGB")
        if not os.path.exists(cond_path):
            os.makedirs(cond_path)

        for j, image in enumerate(images_list[i]):
            # Perform intensity normalization
            normalized_image = exposure.rescale_intensity(image, out_range=(0, 1))
            
            # Convert to uint8 for saving as TIFF
            normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

            # Create a multipage TIFF
            save_path = os.path.join(cond_path, f"{cond}_{j}.tif")
            with Image.fromarray(normalized_image_uint8) as img:
                img.save(save_path)
                print(f"Saved {save_path}")

# Modify your code to call the preprocessing function
if __name__ == "__main__":
    # Images path
    conditions_list = ["Control", "1Fe", "5Fe", "10Fe", "50Fe", "100Fe", "1Ti", "5Ti",
                       "10Ti", "50Ti", "100Ti", "1Sn", "5Sn", "10Sn", "50Sn", "100Sn"]
    files_path = "/Users/marcalbesa/Desktop/TFG/fotos_immunos"
    
    # List to store the loaded images
    images_list = []
    
    # Directory containing TIFF images
    for cond in conditions_list:
        cond_images = []
        path = os.path.join(files_path, f"ML_A549_{cond}_PhaD_20X_RGB")
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith(".tif"):
                    file_path = os.path.join(path, filename)
                    image = plt.imread(file_path)
                    image = resize(image, (256, 256, 3))
                    cond_images.append(image)
        else:
            print(f"Directory does not exist: {path}")
        images_list.append(cond_images)
    
    # Preprocess and save images
    output_path = "/Users/marcalbesa/Desktop/TFG/fotos_immunos_norm"
    preprocess_and_save(images_list, conditions_list, output_path)
