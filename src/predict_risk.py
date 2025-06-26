import torch
from risk_predictor import RiskPredictor
from data_loader import MammogramDataset
from config_tuning import config

def load_risk_model(model_path, num_birads=6, attr_dim=1, embed_dim=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskPredictor(num_birads=num_birads, attr_dim=attr_dim, embed_dim=embed_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict_birads(model, image, attributes):
    device = next(model.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
    if attributes.dim() == 1:
        attributes = attributes.unsqueeze(0)
    image = image.to(device)
    attributes = attributes.to(device)
    logits = model(image, attributes)
    pred = logits.argmax(dim=1).item()    
    return pred  # index of the predicted BIRADS class

# Usage example
if __name__ == "__main__":
    # Load model
    model_path = "./models/risk_predictor_epoch150.pth"
    model = load_risk_model(model_path, num_birads=6, attr_dim=1, embed_dim=128)

    # Load a sample from the test set
    data_dir = config.data_dir
    _, test_loader = MammogramDataset.get_data_loaders(data_dir, batch_size=1)
    sample = next(iter(test_loader))
    image = sample["image"].squeeze(0)           # [C, H, W]
    attributes = sample["attributes"][:1]        # only breast_density (adjust if you want more attributes)

    # Predict BIRADS
    pred_birads = predict_birads(model, image, attributes)
    print(f"BIRADS predicted: {pred_birads}")
