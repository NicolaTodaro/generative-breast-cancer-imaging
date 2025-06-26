# src/config_tuning.py

class Config:
    # Model parameters
    latent_dim = 128
    input_shape = (1, 224, 224)  # example for grayscale mammograms, adjust as needed
    
    # Attributi anatomici: 4 (density) + 2 (view)
    attr_dim = 16

    # Transformer parameters (opzionali)
    num_layers = 2
    num_heads = 2

    # Training parameters
    batch_size = 8
    learning_rate = 5e-4
    epochs = 150

    # Diffusion or VAE-specific parameters
    diffusion_steps = 100
    beta_start = 1e-4
    beta_end = 0.02

    # Add paths for saving models and logs
    model_save_path = "./models/"
    log_dir = "./logs/"

# Optionally, instantiate a global configuration object
config = Config()
