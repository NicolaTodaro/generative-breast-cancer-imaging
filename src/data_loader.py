import os
import re
import zipfile as zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pydicom
from PIL import Image, ImageDraw
from skimage.transform import resize


"""
PyTorch Dataset for mammography images in DICOM format, supporting masks, metadata, and robust attribute parsing.

Loads images and annotations from a zip archive structured as VINDR-MAMMO_DATASET.
Extracted anatomical attributes include density, view, category, BIRADS, laterality, and region coordinates.

Attributes:
    data_dir (str): Path to the zip archive containing images and CSV.
    transform (callable): Transformations to apply to images.
    use_masks (bool): If True, generates masks for the images.
    use_metadata (bool): If True, extracts anatomical attributes.
    category_list (list): Sorted list of all category combinations.
    category_map (dict): Mapping from category combination to index.
    density_map (dict): Mapping from density letter to index.
    view_map (dict): Mapping from view type to index.

Each item returned by __getitem__ is a dictionary with:
    - image: normalized tensor [1, H, W]
    - mask: binary tensor [1, H, W]
    - masked_image: masked image tensor [1, H, W]
    - attributes: tensor of indices and coordinates [9]
    - density, view, split, category, birads: original metadata strings

Example:
    dataset = MammogramDataset("data.zip")
    sample = dataset[0]
    image = sample["image"]
    attributes = sample["attributes"]
"""

class MammogramDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_masks=True, use_metadata=True):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.use_masks = use_masks
        self.use_metadata = use_metadata

        # Open zip file and load CSV
        with zip.ZipFile(self.data_dir, 'r') as zip_ref:
            with zip_ref.open('VINDR-MAMMO_DATASET/finding_annotations.csv') as csv_file:
                self.metadata = pd.read_csv(csv_file)
            # Save the list of all DICOM files in the zip
            self.dicom_files = [f for f in zip_ref.namelist() if f.endswith('.dicom')]
            # Get the image_id actually present in the zip
            available_ids = set([os.path.splitext(os.path.basename(f))[0] for f in self.dicom_files])
            # Filter the metadata to keep only those present in the zip
            self.metadata = self.metadata[self.metadata["image_id"].isin(available_ids)].reset_index(drop=True)

        all_categories = set()
        for cats in self.metadata["finding_categories"].dropna():
            combo = ",".join(sorted([c.strip() for c in cats.split(",")]))
            all_categories.add(combo)
        # Sort categories and create a mapping
        self.category_list = sorted(list(all_categories))
        self.category_map = {cat: idx for idx, cat in enumerate(self.category_list)}
        # Map breast density to indices
        self.density_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        # Map view positions to indices
        self.view_map = {"CC": 0, "MLO": 1}
        
    def __len__(self):
        return len(self.metadata)

    def find_dicom_path(self, image_id):
        #Find the DICOM file path based on image_id
        for dicom_path in self.dicom_files:
            if dicom_path.endswith(f"{image_id}.dicom"):
                return dicom_path
        raise FileNotFoundError(f"DICOM file for image_id {image_id} not found.")

    def __getitem__(self, idx):
        # Get attributes from metadata
        row = self.metadata.iloc[idx] 
        image_id = row["image_id"]
        density_str = row["breast_density"] if pd.notna(row["breast_density"]) else "N/A"
        view_str = row["view_position"] if pd.notna(row["view_position"]) else "N/A"
        category_str = row["finding_categories"] if pd.notna(row["finding_categories"]) else "N/A"
        birads_str = row["breast_birads"] if pd.notna(row["breast_birads"]) else "N/A"
        split_str = row["split"] if pd.notna(row["split"]) else "N/A"
        
        dcm_path = self.find_dicom_path(image_id) 
        with zip.ZipFile(self.data_dir, 'r') as zip_ref:
            with zip_ref.open(dcm_path) as dicom_file:
                dcm = pydicom.dcmread(dicom_file)
                image_array = dcm.pixel_array.astype(np.float32)
                #  Normalize the pixel values to [0, 255]
                image_array = 255 * (image_array - np.min(image_array)) / (np.ptp(image_array) + 1e-8)
                image = Image.fromarray(image_array.astype(np.uint8)).convert("L")

                # Generate the mask from the coordinates
                mask = None
                if self.use_masks:
                    mask = Image.new("L", image.size, 0)
                    if all(col in row for col in ["xmin", "ymin", "xmax", "ymax"]) and not any(pd.isna(row[c]) for c in ["xmin", "ymin", "xmax", "ymax"]):
                        xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                        draw = ImageDraw.Draw(mask)
                        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

                # Apply transformations 
                image = self.transform(image)
                if mask is not None:
                    mask = self.transform(mask)
                else:
                    mask = torch.zeros_like(image)

                # Create (masked image = image * mask)
                masked_image = image * (mask > 0.5).float() if mask is not None else torch.zeros_like(image)

                # Prepare anatomical attributes (e.g., density, view)
                attributes = torch.tensor([])
                # Prepare anatomical attributes as indices
                if self.use_metadata:
                    # Density: extract letter A/B/C/D
                    density_raw = str(row.get("breast_density", "A"))
                    match = re.search(r"([A-D])", density_raw)
                    density_letter = match.group(1) if match else "A"
                    density_idx = self.density_map.get(density_letter, 0)
                    # View: extract side (L/R) and view (CC/MLO)
                    view_raw = str(row.get("view_position", "CC"))
                    match = re.search(r"([LR])[-_]?(CC|MLO)", view_raw)
                    if match:
                        side = match.group(1)  # 'L' or 'R'
                        view_type = match.group(2)  # 'CC' or 'MLO'
                    else:
                        # fallback: find CC or MLO
                        match = re.search(r"(CC|MLO)", view_raw)
                        side = "L"  # default
                        view_type = match.group(1) if match else "CC"
                    view_idx = self.view_map.get(view_type, 0)
                    # side attribute
                    side_idx = 0 if side == "L" else 1

                    # Use sorted combination of categories as a single class
                    cats = str(row.get("finding_categories", ""))
                    combo = ",".join(sorted([c.strip() for c in cats.split(",") if c.strip() in self.category_map]))
                    category_idx = self.category_map.get(combo, 0)
                    # BIRADS: extract digit from string, default to 0 if not found
                    birads_raw = str(row.get("breast_birads", "0"))
                    match = re.search(r"(\d+)", birads_raw)
                    birads_digit = match.group(1) if match else "0"
                    birads_idx = ["0", "1", "2", "3", "4", "5"].index(birads_digit)                    
                    # Coordinates: use -1 if missing
                    xmin = int(row["xmin"]) if "xmin" in row and not pd.isna(row["xmin"]) else -1
                    ymin = int(row["ymin"]) if "ymin" in row and not pd.isna(row["ymin"]) else -1
                    xmax = int(row["xmax"]) if "xmax" in row and not pd.isna(row["xmax"]) else -1
                    ymax = int(row["ymax"]) if "ymax" in row and not pd.isna(row["ymax"]) else -1
                    
                    attributes = torch.tensor([density_idx, view_idx, category_idx, birads_idx, side_idx, xmin, ymin, xmax, ymax], dtype=torch.long)
                else:
                    attributes = torch.tensor([], dtype=torch.long)
        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "attributes": attributes,
            "density": density_str,
            "view": view_str,
            "split": split_str,
            "category": category_str,
            "birads": birads_str, 
        }
    
    @staticmethod
    def denormalize(tensor, mean=0.5, std=0.5):
        return tensor * std + mean

    @staticmethod
    def show_sample(sample):
        image = MammogramDataset.denormalize(sample["image"].squeeze().cpu()).numpy()
        mask = sample["mask"].squeeze().cpu().numpy()
        masked_image = MammogramDataset.denormalize(sample["masked_image"].squeeze().cpu()).numpy()
        
        #If the mask is all zeros, just show the image
        coords = np.argwhere(mask > 0.5)
        if np.sum(mask) == 0 or coords.size == 0:
            plt.imshow(image, cmap='gray')
            plt.title(f"Image\nDensity: {sample['density']}, View: {sample['view']}")
            plt.axis('off')
            print("NO MASK! Nothing to highlight in here.")
            plt.show()
            return
        
        # Find the bounding box of the mask
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
                
        # 1. Original image
        # 2. Image with yellow rectangle
        # 3. Cropped region resized to fit within a 128x128 canvas
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Image")
        axs[0].axis('off')

        # Highlight the mask on the original image
        axs[1].imshow(image, cmap='gray')
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                             edgecolor='yellow', facecolor='none', linewidth=2)
        axs[1].add_patch(rect)
        axs[1].set_title("Image with Mask Highlighted")
        axs[1].axis('off')

        # Crop the image using the mask coordinates        
        cropped = image[y_min:y_max, x_min:x_max]
        canvas_size = 128
        cropped_h, cropped_w = cropped.shape

        if cropped_h > canvas_size or cropped_w > canvas_size:
            # Resize the crop to fit within the canvas size 
            scale = min(canvas_size / cropped_h, canvas_size / cropped_w)
            new_h, new_w = int(cropped_h * scale), int(cropped_w * scale)
            cropped_resized = resize(cropped, (new_h, new_w), preserve_range=True)
        else:
            cropped_resized = cropped

        canvas = np.zeros((canvas_size, canvas_size), dtype=cropped.dtype)
        y_offset = (canvas_size - cropped_resized.shape[0]) // 2
        x_offset = (canvas_size - cropped_resized.shape[1]) // 2
        canvas[y_offset:y_offset+cropped_resized.shape[0], x_offset:x_offset+cropped_resized.shape[1]] = cropped_resized

        axs[2].imshow(canvas, cmap='gray')
        axs[2].set_title("Masked (zoom)")
        axs[2].axis('off')
        plt.suptitle(f"Density: {sample['density']}, View: {sample['view']}, Category: {sample['category']}, BIRADS: {sample['birads']}, Split: {sample['split']}")
        plt.show()

    @staticmethod
    def from_split(data_dir, split_value, transform=None, use_masks=True, use_metadata=True):
        #Dataset factory method to create a dataset for a specific split
        dataset = MammogramDataset(data_dir, transform, use_masks, use_metadata)
        flag = dataset.metadata["split"] == split_value
        dataset.metadata = dataset.metadata[flag].reset_index(drop=True)
        return dataset

    @staticmethod
    def get_data_loaders(data_dir, batch_size, transform=None, use_masks=True, use_metadata=True):
        #Factory method to get train and test data loaders
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        train_dataset = MammogramDataset.from_split(data_dir, "training", transform, use_masks, use_metadata)
        test_dataset = MammogramDataset.from_split(data_dir, "test", transform, use_masks, use_metadata)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

