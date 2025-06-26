# src/utils.py

import torch
import matplotlib.pyplot as plt

def display_images(real, reconstructed, n=5):
    """
    Display a few real vs. reconstructed images side by side.
    """
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Real image
        plt.subplot(2, n, i + 1)
        plt.imshow(real[i].squeeze(), cmap="gray")
        plt.title("Real")
        plt.axis("off")
        
        # Reconstructed image
        plt.subplot(2, n, n + i + 1)
        plt.imshow(reconstructed[i].squeeze(), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)
