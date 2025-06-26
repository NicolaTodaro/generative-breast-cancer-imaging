import torch
from PIL import Image
from config_tuning import config
from model import NewLatentDiffusionTransformer as LatentDiffusionModel
from data_loader import MammogramDataset
import torchvision.transforms as transforms


def load_model(model_path, data_dir):
    dataset = MammogramDataset(data_dir)
    category_list = dataset.category_list
    model = LatentDiffusionModel(config, category_list)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

def generate_image(model, attributes, image_shape=None):
    device = next(model.parameters()).device  # Get the device of the model parameters
    with torch.no_grad():
        if image_shape is None:
            image_shape = (1, ) + tuple(config.input_shape)
        image = torch.randn(image_shape, device=device)
        # If there is at least one active category (indices 6:10)
        if attributes[0, 6:10].sum() > 0:
            masked_image = torch.randn_like(image)
        else:
            masked_image = torch.zeros_like(image)
        attributes = attributes.to(device)
        recon, _ = model(image, masked_image, attributes)
        # Denormalize the image
        recon = MammogramDataset.denormalize(recon).clamp(0, 1)
    return recon