# src/compare_model.py

import torch
import numpy as np
from config_tuning import config
from data_loader import MammogramDataset
from model import NewLatentDiffusionTransformer as LatentDiffusionModel
from metrics import compute_ssim, compute_psnr

def evaluate_model(model, data_loader):
    """
    Evaluate a given model on the provided data loader by computing average SSIM and PSNR.
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation.
        
    Returns:
        tuple: Average SSIM and PSNR scores over the dataset.
    """
    model.eval()
    ssim_scores = []
    psnr_scores = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"]
            masked_images = batch.get("masked_image", None)
            attributes = batch.get("attributes", None)
            recon, _ = model(images, masked_images, attributes)
            # Iterate over each image in the batch
            for i in range(images.size(0)):
                real = images[i].squeeze().cpu().numpy()
                fake = recon[i].squeeze().cpu().numpy()
                ssim_scores.append(compute_ssim(real, fake))
                psnr_scores.append(compute_psnr(real, fake))
    
    return np.mean(ssim_scores), np.mean(psnr_scores)

def compare_models(model_paths, test_loader):
    """
    Compare multiple models and output their average SSIM and PSNR scores.
    
    Parameters:
        model_paths (dict): A dictionary where keys are model names and values are paths to the model checkpoints.
        data_dir (str): Path to the directory containing evaluation/test data.
        
    Returns:
        dict: Dictionary containing evaluation metrics for each model.
    """

    results = {}
    category_list = test_loader.dataset.category_list

    for label, path in model_paths.items():
        # Initialize model and load weights
        model = LatentDiffusionModel(config, category_list)
        model.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        ssim, psnr = evaluate_model(model, test_loader)
        results[label] = {'SSIM': ssim, 'PSNR': psnr}
    
    return results

if __name__ == "__main__":
    # Example usage: define the models to compare
    # Ensure that these checkpoint paths exist in your repository.
    compressed_dataset = '/content/drive/MyDrive/DATASET-BREAST_CANCER/VINDR-MAMMO_DATASET.zip'
    _, test_loader = MammogramDataset.get_data_loaders(compressed_dataset, batch_size=8)

    model_paths = {
        "Baseline Model": "./models/baseline.pth",
        "Diffusion Model": "./models/diffusion_epoch50.pth",
    }
    
    # Also ensure your data_dir contains test images processed in the same way as during training.
    results = compare_models(model_paths, test_loader)
    
    for label, metrics in results.items():
        print(f"Model: {label}")
        print(f"\tAverage SSIM: {metrics['SSIM']:.4f}")
        print(f"\tAverage PSNR: {metrics['PSNR']:.2f} dB")
